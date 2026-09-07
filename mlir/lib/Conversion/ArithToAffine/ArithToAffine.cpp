#include "mlir/Conversion/ArithToAffine/ArithToAffine.h"

#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/PresburgerSpace.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DynamicAPInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTARITHTOAFFINE
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::presburger;

namespace {

using IntVector = SmallVector<llvm::DynamicAPInt>;

// A lower/upper bound pair on the signed interpretation of a value.
using Bound = std::pair<DynamicAPInt, DynamicAPInt>;

// The definition of a local variable introduced for a division
// or modulus. The variable represents `nom / den`, where
// `nom` is a coefficient vector over the symbols and the local variables defined *before* this one.
struct LocalDef {
  IntVector nom;
  DynamicAPInt den;
};

struct ArithToAffinePass : impl::ConvertArithToAffineBase<ArithToAffinePass> {
  using Base::Base;

  struct AffineMapResult {
    SmallVector<Value> operands;
    AffineExpr expr;
    IntegerRelation constraint;
    // Definitions of the local variables of `constraint`, in column order.
    SmallVector<LocalDef> locals;
  };

  struct AffineResult {
    // The affine expression from symbols to the value.
    AffineExpr expr;
    // The constraints on the symbols such that intermediate results will
    // not overflow, under the signed interpretation. Division and modulus
    // subexpressions are represented with local variables.
    IntegerRelation constraint;
    // Facts known to hold about the symbols unconditionally (e.g. from
    // `nsw`/`nuw` flags, whose violation would make the operation poison),
    // over the same space as `constraint`. Rows of `constraint` implied by
    // these facts do not need to be checked at runtime.
    IntegerRelation reference;
    // Definitions of the local variables of `constraint` and `reference`,
    // in column order.
    SmallVector<LocalDef> locals;
  };

  struct LiftResult {
    Value apply;
    Value condition;
  };

  // Maps each affine value to its dimension index in a presburger relation.
  using DimIndex = llvm::DenseMap<Value, int>;

  std::optional<AffineMapResult> valueToAffine(Value value);
  std::optional<AffineResult> constructAffineMap(Value value,
                                                 const DimIndex &dimIndex);
  AffineMapResult tidyResult(const AffineResult &result, Value value,
                             ValueRange operands);
  // Adds rows to `rel` bounding the variable at position `pos`, which
  // corresponds to `value`, based on the integer range analysis result.
  void addAnalyzedRangeConstraints(Value value, unsigned pos,
                                   IntegerRelation &rel) const;
  // Adds rows to `reference` bounding the symbol at position `pos`, which
  // corresponds to `value`, based on bounds computed via
  // ValueBoundsOpInterface (e.g. the bounds of an scf.for induction variable
  // in terms of the loop bounds). `valueToSymbol` maps each lifted operand
  // to its symbol position in `reference`.
  void addValueBoundsConstraints(Value value, unsigned pos,
                                 const DenseMap<Value, unsigned> &valueToSymbol,
                                 IntegerRelation &reference) const;
  // Adds rows to `reference` recording the conditions of the `scf.if` ops
  // whose regions contain all uses of `value`, when those conditions are
  // conjunctions of affine comparisons over the lifted operands (then
  // branch), or disjunctions of them whose negation is again a conjunction
  // (else branch). `valueToSymbol` maps each lifted operand to its symbol
  // position in `reference`.
  void addBranchConditionConstraints(
      Value value, const DenseMap<Value, unsigned> &valueToSymbol,
      IntegerRelation &reference);
  // Adds the conjuncts of `cond` to `reference`. When `negated`, `cond` is
  // known to be false instead, so only its disjuncts survive negation as
  // conjuncts. Anything that is not an and/or (as appropriate) of signed
  // affine comparisons is silently ignored.
  void addConditionConstraints(Value cond, bool negated,
                               const DenseMap<Value, unsigned> &valueToSymbol,
                               IntegerRelation &reference);
  void getDependentDialects(DialectRegistry &registry) const override {
    impl::ConvertArithToAffineBase<ArithToAffinePass>::getDependentDialects(
        registry);
    scf::registerValueBoundsOpInterfaceExternalModels(registry);
  }
  // Computes the offset and represent it as an affine map for `value`.
  LiftResult lift(Value value);
  // Synthesizes a runtime check for `rel` over `values` (the symbols), where
  // `bounds` gives a sound signed range for each symbol value.
  Value synthesizeCheck(OpBuilder &builder, Location loc,
                        const IntegerRelation &rel,
                        ArrayRef<LocalDef> localDefs, ValueRange values,
                        ArrayRef<Bound> bounds);
  // Returns sound signed bounds for `value`: its analyzed integer range if
  // available, otherwise the bounds of its type (64 bits for index).
  Bound bounds(Value value) const;
  // Lifts the address of `load` individually, guarding the lifted address
  // with a per-load runtime check.
  void liftLoadAddress(LLVM::LoadOp load);
  // Intersects the lifting constraints of all loads directly inside `loop`
  // and emits a single check before it:
  //   if (check) { lifted loop } else { original loop }
  // Falls back to per-load lifting when the constraints cannot be hoisted.
  void liftLoop(Operation *loop);
  // Computes a bound of `inner` (a symbol defined inside `loop`) in terms of
  // values defined outside `loop`, expressed as coefficients over the
  // hoisted symbols. Bound operands defined outside the loop are registered
  // in `hoistedOperands`/`hoistedIndex` on demand.
  std::optional<IntVector>
  computeOuterBound(Value inner, BoundType type, Operation *loop,
                    SmallVectorImpl<Value> &hoistedOperands,
                    DenseMap<Value, unsigned> &hoistedIndex);
  void runOnOperation() override;

private:
  mlir::DataFlowSolver *solver;
};

// We eagerly collect operands up to the point where we either
// cannot push further, or it is unnecessary to do so (when we hit a symbol).
void collectOperandsUpChain(Value value, SmallVector<Value> &operands) {
  Operation *def = value.getDefiningOp();
  if (!def || def->getNumOperands() == 0) {
    operands.push_back(value);
    return;
  }

  if (isa<arith::ArithDialect>(def->getDialect()) ||
      isa<LLVM::GEPOp, LLVM::AddOp, LLVM::SubOp, LLVM::MulOp>(def)) {
    for (auto operand : def->getOperands())
      collectOperandsUpChain(operand, operands);
  } else {
    operands.push_back(value);
  }
}

// Constructs a coefficient vector for the symbols involved in `expr`.
//
// Division and modulus with a positive constant divisor are supported when
// `rel` is provided: a local variable representing the floordiv
// result is appended to `rel`, and its definition is recorded in `locals`.
// When we meet such expressions without `rel`, we directly return nullopt.
std::optional<IntVector>
extractCoefficients(AffineExpr expr, unsigned numSymbols,
                    IntegerRelation *rel = nullptr,
                    SmallVectorImpl<LocalDef> *locals = nullptr) {
  // The current row size of coefficient vectors.
  auto width = [&]() {
    return numSymbols + (rel ? rel->getNumLocalVars() : 0) + 1;
  };

  // Extends a coefficient vector to the current width, inserting zeros
  // before the constant term.
  auto pad = [&](IntVector &vec) {
    DynamicAPInt constant = vec.back();
    unsigned oldWidth = vec.size();
    vec.resize(width());
    vec[oldWidth - 1] = DynamicAPInt(0);
    vec.back() = constant;
  };

  if (auto c = dyn_cast<AffineConstantExpr>(expr)) {
    IntVector result(width());
    result.back() += c.getValue();
    return result;
  }

  if (auto s = dyn_cast<AffineSymbolExpr>(expr)) {
    IntVector result(width());
    result[s.getPosition()] += 1;
    return result;
  }

  if (auto bin = dyn_cast<AffineBinaryOpExpr>(expr)) {
    switch (bin.getKind()) {
    case AffineExprKind::Add: {
      auto l = extractCoefficients(bin.getLHS(), numSymbols, rel, locals);
      auto r = extractCoefficients(bin.getRHS(), numSymbols, rel, locals);
      if (!l || !r)
        return std::nullopt;

      pad(*l);
      for (unsigned i = 0; i < r->size(); i++)
        (*l)[i] += (*r)[i];

      return l;
    }

    case AffineExprKind::Mul: {
      AffineExpr other;
      DynamicAPInt constant(0);
      if (auto c = dyn_cast<AffineConstantExpr>(bin.getLHS()))
        other = bin.getRHS(), constant = c.getValue();
      else if (auto c = dyn_cast<AffineConstantExpr>(bin.getRHS()))
        other = bin.getLHS(), constant = c.getValue();
      else
        return std::nullopt;

      auto coeffs = extractCoefficients(other, numSymbols, rel, locals);
      if (!coeffs)
        return std::nullopt;
      for (auto &coeff : *coeffs)
        coeff *= constant;
      return coeffs;
    }

    case AffineExprKind::FloorDiv:
    case AffineExprKind::CeilDiv:
    case AffineExprKind::Mod: {
      if (!rel)
        return std::nullopt;
      auto div = dyn_cast<AffineConstantExpr>(bin.getRHS());
      if (!div || div.getValue() <= 0)
        return std::nullopt;

      DynamicAPInt den(div.getValue());
      auto nom =
          extractCoefficients(bin.getLHS(), numSymbols, rel, locals);
      if (!nom)
        return std::nullopt;
      // Convert ceildiv to floordiv:
      //   ceil(a / b) = floor((a + b - 1) / b)
      if (bin.getKind() == AffineExprKind::CeilDiv)
        nom->back() += den - 1;

      // Index of the new local variable we introduce.
      std::optional<unsigned> index;
      if (locals) {
        // Reuse an existing local variable if it has the same definition.
        for (unsigned i = 0; i < locals->size() && !index; i++) {
          const LocalDef &def = (*locals)[i];
          if (def.den != den)
            continue;
          // def.den is over the symbols and locals 0..i-1, while
          // `den` is over the symbols and all current locals.
          unsigned n = def.nom.size();
          bool equal = nom->back() == def.nom.back();
          for (unsigned k = 0; k + 1 < n && equal; k++)
            equal = (*nom)[k] == def.nom[k];
          for (unsigned k = n - 1; k + 1 < nom->size() && equal; k++)
            equal = (*nom)[k] == 0;
          if (equal)
            index = numSymbols + i;
        }
      }
      if (!index) {
        index = rel->addLocalFloorDiv(*nom, den);
        if (locals)
          locals->push_back({*nom, den});
      }

      if (bin.getKind() != AffineExprKind::Mod) {
        IntVector result(width());
        result[*index] += 1;
        return result;
      }
      // Convert modulus to floordiv:
      //   a % b = a - b * (a / b).
      pad(*nom);
      (*nom)[*index] -= den;
      return nom;
    }
    default:
      return std::nullopt;
    }
  }

  return std::nullopt;
}

template <class T>
T pow(T x, unsigned y) {
  T result(1);
  while (y > 0) {
    if (y % 2)
      result *= x;
    x *= x;
    y /= 2;
  }
  return result;
}

std::pair<DynamicAPInt, DynamicAPInt> getSignedBounds(Type type) {
  assert(type.isInteger());
  unsigned width = type.getIntOrFloatBitWidth();
  auto power = pow(DynamicAPInt(2), width - 1);
  return {-power, power - 1};
}

// The bit width of a value used in check arithmetic; index values are
// treated as 64-bit.
unsigned valueWidth(Value value) {
  if (isa<IndexType>(value.getType()))
    return 64;
  return cast<IntegerType>(value.getType()).getWidth();
}

// Returns whether `value` is defined inside `loop`, including block
// arguments of the regions of `loop`.
bool valueInLoop(Value value, Operation *loop) {
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    Operation *parent = arg.getOwner()->getParent()->getParentOp();
    return parent == loop || loop->isAncestor(parent);
  }
  Operation *def = value.getDefiningOp();
  return def && loop->isAncestor(def);
}

// Normal division truncates towards zero.
// For floor division, we must subtract 1 when x < 0.
DynamicAPInt floordiv(const DynamicAPInt &x, const DynamicAPInt &y) {
  auto q = x / y;
  return x < 0 ? q - 1 : q;
}

// Shift elements starting from `n` by `shift` amount to the right. 
IntVector shift(const IntVector &vec, unsigned n, unsigned shift) {
  IntVector result(vec.size() + shift);
  for (unsigned i = 0; i < n; i++)
    result[i] = vec[i];
  for (unsigned i = n; i + 1 < vec.size(); i++)
    result[i + shift] = vec[i];
  result.back() = vec.back();
  return result;
}

// Merges the local variable definitions of two intersected relations,
// shifting the local columns referenced by `b`'s definitions to match.
SmallVector<LocalDef> mergeLocalDefs(SmallVector<LocalDef> a,
                                     const SmallVector<LocalDef> &b,
                                     unsigned numSymbols) {
  unsigned numLocalsA = a.size();
  for (const LocalDef &def : b)
    a.push_back({shift(def.nom, numSymbols, numLocalsA), def.den});
  return a;
}

// Returns whether `row` is a positive multiple of `pattern`, i.e., whether
// the two are logically equivalent as `>= 0` constraints.
static bool isPositiveMultiple(ArrayRef<DynamicAPInt> row,
                               ArrayRef<DynamicAPInt> pattern) {
  unsigned pivot = 0;
  while (pivot < pattern.size() && pattern[pivot] == 0)
    pivot++;
  if (pivot == pattern.size())
    return false;
  if (row[pivot] * pattern[pivot] <= 0)
    return false;
  for (unsigned j = 0; j < pattern.size(); j++)
    if (row[j] * pattern[pivot] != pattern[j] * row[pivot])
      return false;
  return true;
}

LogicalResult populateConstraint(AffineExpr expr, unsigned numSymbols,
                                 Type type, IntegerRelation &constraint,
                                 SmallVectorImpl<LocalDef> &localDefs) {
  auto maybeSum =
      extractCoefficients(expr, numSymbols, &constraint, &localDefs);
  if (!maybeSum)
    return failure();

  IntVector result = *maybeSum;
  // We must guarantee that the result does not overflow:
  //   min <= result <= max.
  auto [min, max] = getSignedBounds(type);
  // IntegerRelations expect constraints of form `row >= 0`.
  // First add a row for `result - min >= 0`.
  result.back() -= min;
  constraint.addInequality(result);
  // Then add a row for `result - sum >= 0`.
  // Recover the last element for `sum` and flip all signs.
  result.back() += min;
  for (auto &e : result)
    e *= -1;
  result.back() += max;
  constraint.addInequality(result);
  // `simplify()` may drop rows that are rationally implied by the others,
  // including the defining rows of local variables. Restore them: they hold
  // by construction, and they are needed for the local variables to be
  // recognized as divisions (e.g. when merging relations with duplicate
  // divisions).
  // Therefore, we cannot call simplify() here.
  return success();
}

// Replays the definitions of the local variables `localDefs` that `rel`
// does not have yet, so that its local variables are exactly those described
// by `localDefs`, with the same defining constraints.
void syncLocalDefs(IntegerRelation &rel,
                   const SmallVectorImpl<LocalDef> &localDefs) {
  while (rel.getNumLocalVars() < localDefs.size()) {
    const LocalDef &def = localDefs[rel.getNumLocalVars()];
    rel.addLocalFloorDiv(def.nom, def.den);
  }
}

// Adds rows to `rel` recording the known fact `lower <= expr <= upper`,
// introducing local variables for divisions as needed. Local variables
// introduced into `rel` are replayed into `other` as well, so that the two
// relations remain in the same space.
//
// Silently fails if `expr` is not representible in an integer relation.
void addKnownBounds(AffineExpr expr, unsigned numSymbols,
                    const DynamicAPInt &lower, const DynamicAPInt &upper,
                    IntegerRelation &rel, IntegerRelation &other,
                    SmallVectorImpl<LocalDef> &localDefs) {
  auto maybeSum = extractCoefficients(expr, numSymbols, &rel, &localDefs);
  if (!maybeSum)
    return;
  syncLocalDefs(other, localDefs);

  IntVector result = *maybeSum;
  // Row for `result - lower >= 0`.
  result.back() -= lower;
  rel.addInequality(result);
  // Row for `upper - result >= 0`.
  result.back() += lower;
  for (auto &e : result)
    e *= -1;
  result.back() += upper;
  rel.addInequality(result);
}

// Adds the constraint `expr >= 0` to `constraint`, introducing local
// variables for divisions/moduli as needed. The local variables are replayed
// into `reference` as well, so that the two relations remain in the same
// space.
//
// Silently fails if `expr` is not representible in an integer relation.
LogicalResult addNonNegativeConstraint(AffineExpr expr, unsigned numSymbols,
                                       IntegerRelation &constraint,
                                       IntegerRelation &reference,
                                       SmallVectorImpl<LocalDef> &localDefs) {
  auto coeffs = extractCoefficients(expr, numSymbols, &constraint, &localDefs);
  if (!coeffs)
    return failure();
  syncLocalDefs(reference, localDefs);
  constraint.addInequality(*coeffs);
  return success();
}

int log2(DynamicAPInt x) {
  int v = 0;
  while (x != 0) {
    x /= 2;
    v++;
  }
  return v;
}

// Returns the exponent `e` such that `x == 2^e` when `x` is a positive power
// of two, and nullopt otherwise.
std::optional<int> exactLog2(const DynamicAPInt &x) {
  if (x < 1)
    return std::nullopt;
  DynamicAPInt v(x);
  int e = 0;
  while (v % 2 == 0)
    v /= 2, e++;
  return v == 1 ? std::optional<int>(e) : std::nullopt;
}

Value ArithToAffinePass::synthesizeCheck(OpBuilder &builder, Location loc,
                                         const IntegerRelation &rel,
                                         ArrayRef<LocalDef> localDefs,
                                         ValueRange inValues,
                                         ArrayRef<Bound> inBounds) {
  assert(inValues.size() == inBounds.size() &&
         "check values and bounds out of sync");
  assert(rel.getNumLocalVars() == localDefs.size() &&
         "local variable definitions out of sync with constraint");

  llvm::errs() << "synthesizing:\n";
  rel.dump();

  // The values of the columns of `rel`: the symbols followed by the local
  // variables materialized from their definitions.
  SmallVector<Value> values(inValues.begin(), inValues.end());
  SmallVector<Bound> bounds(inBounds.begin(), inBounds.end());
  values.reserve(inValues.size() + localDefs.size());
  bounds.reserve(inValues.size() + localDefs.size());

  unsigned numSymbols = values.size();
  unsigned numLocals = localDefs.size();

  // Check whether `row` is defining one of the LocalDefs.
  const auto isDefiningRow = [&](ArrayRef<DynamicAPInt> row) {
    for (unsigned i = 0; i < numLocals; i++) {
      const LocalDef &def = localDefs[i];
      IntVector lower(numSymbols + numLocals + 1);
      for (unsigned j = 0; j + 1 < def.nom.size(); j++)
        lower[j] = def.nom[j];
      lower[numSymbols + i] = -def.den;
      lower.back() = def.nom.back();
      if (isPositiveMultiple(row, lower))
        return true;
      // divisor * q_i + (divisor - 1) - dividend >= 0.
      IntVector upper(numSymbols + numLocals + 1);
      for (unsigned j = 0; j + 1 < def.nom.size(); j++)
        upper[j] = -def.nom[j];
      upper[numSymbols + i] = def.den;
      upper.back() = def.den - 1 - def.nom.back();
      if (isPositiveMultiple(row, upper))
        return true;
    }
    return false;
  };

  // Determine the local variables that really needs to be checked.
  // If a variable is only involved in its defining row, then we
  // don't need to emit it.
  SmallVector<bool> needed(numLocals, false);
  for (unsigned i = 0, e = rel.getNumInequalities(); i < e; i++) {
    auto row = rel.getInequality(i);
    if (isDefiningRow(row))
      continue;
    for (unsigned k = 0; k < numLocals; k++)
      if (row[numSymbols + k] != 0)
        needed[k] = true;
  }
  // A local variable might be involved in nominator of locals after it.
  // In this case it is also needed.
  for (unsigned i = numLocals; i > 0; i--)
    if (needed[i - 1])
      for (unsigned k = 0; k < i - 1; k++)
        if (localDefs[i - 1].nom[numSymbols + k] != 0)
          needed[k] = true;

  // The minimum and maximum of `coeffs * values + constant` given the bounds.
  const auto rangeOf = [&](ArrayRef<DynamicAPInt> coeffs,
                           const DynamicAPInt &constant) {
    DynamicAPInt min(constant), max(constant);
    for (unsigned j = 0; j < coeffs.size(); j++) {
      min += coeffs[j] * (coeffs[j] < 0 ? bounds[j].second : bounds[j].first);
      max += coeffs[j] * (coeffs[j] < 0 ? bounds[j].first : bounds[j].second);
    }
    return std::pair{min, max};
  };
  // The bit width needed to represent `coeffs * values + constant`.
  const auto widthFor = [&](ArrayRef<DynamicAPInt> coeffs,
                            const DynamicAPInt &min, const DynamicAPInt &max) {
    int bitWidth = 1 + std::max(log2(min), log2(max));
    for (unsigned j = 0; j < coeffs.size(); j++)
      if (coeffs[j] != 0)
        bitWidth = std::max(bitWidth, 1 + (int)valueWidth(values[j]));
    return bitWidth;
  };
  // Emits `coeffs * values + constant` in `intType`.
  const auto emitLinearForm = [&](ArrayRef<DynamicAPInt> coeffs,
                                  const DynamicAPInt &constant, Type intType) {
    unsigned bitWidth = intType.getIntOrFloatBitWidth();
    Value v = arith::ConstantIntOp::create(
        builder, loc, intType,
        llvm::APInt(bitWidth, (int64_t)constant, /*isSigned=*/true));
    for (unsigned j = 0; j < coeffs.size(); j++) {
      if (coeffs[j] == 0)
        continue;
      auto cst = arith::ConstantIntOp::create(
          builder, loc, intType,
          llvm::APInt(bitWidth, (int64_t)coeffs[j], /*isSigned=*/true));
      Value ext;
      if (isa<IndexType>(values[j].getType()))
        ext = arith::IndexCastOp::create(builder, loc, intType, values[j]);
      else
        ext = arith::ExtSIOp::create(builder, loc, intType, values[j]);
      auto mul = arith::MulIOp::create(builder, loc, ext, cst);
      v = arith::AddIOp::create(builder, loc, v, mul);
    }
    return v;
  };

  // Materialize each needed local variable as `nom / den`,
  // where the nom is over the symbols and the locals defined before it.
  for (unsigned i = 0; i < numLocals; i++) {
    if (!needed[i]) {
      values.push_back(Value());
      bounds.push_back({DynamicAPInt(0), DynamicAPInt(0)});
      continue;
    }
    const LocalDef &def = localDefs[i];
    ArrayRef<DynamicAPInt> nom(def.nom);
    auto [min, max] = rangeOf(nom.drop_back(), def.nom.back());
    // The type must additionally hold the (positive) divisor itself.
    int bitWidth =
        std::max(widthFor(nom.drop_back(), min, max), log2(def.den) + 2);
    Type intType = IntegerType::get(builder.getContext(), bitWidth);
    Value v = emitLinearForm(nom.drop_back(), def.nom.back(), intType);
    // An arithmetic right shift implements floor division exactly when the
    // divisor is a power of two. Otherwise, divsi/remsi truncate towards
    // zero and must be corrected to floor division.
    Value q;
    if (std::optional<int> exponent = exactLog2(def.den)) {
      auto shift =
          arith::ConstantIntOp::create(builder, loc, intType, *exponent);
      q = arith::ShRSIOp::create(builder, loc, v, shift);
    } else {
      auto den =
          arith::ConstantIntOp::create(builder, loc, intType, (int64_t)def.den);
      auto zero = arith::ConstantIntOp::create(builder, loc, intType, 0);
      auto one = arith::ConstantIntOp::create(builder, loc, intType, 1);
      q = arith::DivSIOp::create(builder, loc, v, den);
      Value r = arith::RemSIOp::create(builder, loc, v, den);
      Value negRem = arith::CmpIOp::create(builder, loc,
                                           arith::CmpIPredicate::slt, r, zero);
      q = arith::SelectOp::create(
          builder, loc, negRem, arith::SubIOp::create(builder, loc, q, one), q);
    }
    values.push_back(q);
    bounds.push_back({floordiv(min, def.den),
                      floordiv(max, def.den)});
  }

  const auto synthesize = [&](bool isEq) {
    unsigned numRows = isEq ? rel.getNumEqualities() : rel.getNumInequalities();
    Value condition =
        arith::ConstantIntOp::create(builder, loc, builder.getI1Type(), 1);
    for (unsigned i = 0; i < numRows; i++) {
      auto row = isEq ? rel.getEquality(i) : rel.getInequality(i);
      // Rows pinning a local variable to its definition hold by construction.
      if (!isEq && isDefiningRow(row))
        continue;
      ArrayRef<DynamicAPInt> coeffs = row.drop_back();

      // Compute the maximum and minimum value given the bounds.
      auto [min, max] = rangeOf(coeffs, row.back());
      Type intType =
          IntegerType::get(builder.getContext(), widthFor(coeffs, min, max));
      Value v = emitLinearForm(coeffs, row.back(), intType);
      auto zero = arith::ConstantIntOp::create(builder, loc, intType, 0);
      auto comparison = arith::CmpIOp::create(
          builder, loc, arith::CmpIPredicate::sge, v, zero);
      condition = arith::AndIOp::create(builder, loc, condition, comparison);
    }
    return condition;
  };
  Value eqCond = synthesize(true);
  Value ineqCond = synthesize(false);
  return arith::AndIOp::create(builder, loc, eqCond, ineqCond);
}

std::optional<ArithToAffinePass::AffineResult>
ArithToAffinePass::constructAffineMap(Value value, const DimIndex &dimIndex) {
  unsigned numSymbols = dimIndex.size();
  PresburgerSpace space = PresburgerSpace::getSetSpace(numSymbols);
  IntegerRelation universe(space);
  MLIRContext *ctx = value.getContext();

  llvm::APInt stepValue;
  if (matchPattern(value, m_ConstantInt(&stepValue))) {
    if (stepValue.getActiveBits() > 64)
      return std::nullopt;

    AffineExpr expr = getAffineConstantExpr(stepValue.getSExtValue(), ctx);
    return AffineResult{expr, universe, universe, {}};
  }

  // We will not trace upwards from block arguments in `valueToAffine()`.
  Operation *def = value.getDefiningOp();
  if (!def) {
    AffineExpr expr = getAffineSymbolExpr(dimIndex.at(value), ctx);
    return AffineResult{expr, universe, universe, {}};
  }

  // So when we reach here, the value must be an operation result.
  // We only consider operations in the arith dialect.
  //
  // We check for binary operations here.
  if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, LLVM::AddOp, LLVM::SubOp,
          LLVM::MulOp>(def)) {
    auto l = constructAffineMap(def->getOperand(0), dimIndex);
    auto r = constructAffineMap(def->getOperand(1), dimIndex);
    if (!l || !r)
      return std::nullopt;

    AffineExpr expr =
        llvm::TypeSwitch<Operation *, AffineExpr>(def)
            .Case<arith::AddIOp>(
                [&](arith::AddIOp) { return l->expr + r->expr; })
            .Case<arith::SubIOp>(
                [&](arith::SubIOp) { return l->expr - r->expr; })
            .Case<arith::MulIOp>(
                [&](arith::MulIOp) { return l->expr * r->expr; })
            .Case<LLVM::AddOp>([&](LLVM::AddOp) { return l->expr + r->expr; })
            .Case<LLVM::SubOp>([&](LLVM::SubOp) { return l->expr - r->expr; })
            .Case<LLVM::MulOp>([&](LLVM::MulOp) { return l->expr * r->expr; })
            .DefaultUnreachable("constructAffineMap: should be exhaustive!");

    auto constraint = l->constraint.intersect(r->constraint);
    auto reference = l->reference.intersect(r->reference);
    auto locals = mergeLocalDefs(std::move(l->locals), r->locals, numSymbols);
    Type type = def->getResult(0).getType();
    if (failed(populateConstraint(expr, numSymbols, type, constraint, locals)))
      return std::nullopt;
    // Give `reference` the local variables introduced above.
    syncLocalDefs(reference, locals);

    // Record facts implied by the overflow flags of the operation. These
    // hold unconditionally (the result is poison otherwise), so rows of
    // `constraint` implied by them do not need a runtime check.
    bool nsw = false, nuw = false;
    llvm::TypeSwitch<Operation *>(def)
        .Case<arith::AddIOp, arith::SubIOp, arith::MulIOp, LLVM::AddOp,
              LLVM::SubOp, LLVM::MulOp>([&](auto op) {
          nsw = op.hasNoSignedWrap();
          nuw = op.hasNoUnsignedWrap();
        })
        .Default([](Operation *) {});
    if (nsw) {
      // `nsw` guarantees that the result does not signed-wrap.
      auto [min, max] = getSignedBounds(type);
      addKnownBounds(expr, numSymbols, min, max, reference, constraint, locals);
    }
    unsigned width = type.getIntOrFloatBitWidth();
    // The modulus 2^width must fit in an int64_t.
    if (nuw && width < 63) {
      // `nuw` guarantees that the result does not unsigned-wrap. With
      // `u(x) = x mod 2^width` denoting the unsigned interpretation,
      // it suffices that:
      //    0 <= u(a) <op> u(b) < 2^width
      // no matter <op> is +, - or *.
      int64_t mod = pow<int64_t>(2, width);
      AffineExpr modL = l->expr % mod;
      AffineExpr modR = r->expr % mod;
      AffineExpr fact;
      if (isa<arith::AddIOp, LLVM::AddOp>(def))
        fact = modL + modR;
      else if (isa<arith::SubIOp, LLVM::SubOp>(def))
        fact = modL - modR;
      else if (isa<arith::MulIOp, LLVM::MulOp>(def))
        fact = modL * modR;
      else
        llvm_unreachable("constructAffineMap: nuw: should be exhaustive!");
      addKnownBounds(fact, numSymbols, DynamicAPInt(0), DynamicAPInt(mod - 1),
                     reference, constraint, locals);
    }

    return AffineResult{expr, constraint, reference, std::move(locals)};
  }

  if (isa<arith::ExtSIOp, arith::IndexCastOp>(def)) {
    // This never imposes any extra constraint, since we chose our internal
    // representation as signed.
    return constructAffineMap(def->getOperand(0), dimIndex);
  }

  if (auto ext = dyn_cast<arith::ExtUIOp>(def)) {
    auto operand = ext.getIn();
    auto maybeResult = constructAffineMap(operand, dimIndex);
    if (!maybeResult)
      return std::nullopt;

    // Under speculation, we require the operand to be non-negative instead of
    // introducing a modulo: the signed and unsigned interpretations of a
    // non-negative value coincide, so the zero extension does not change the
    // value.
    if (speculateNoModulus) {
      if (failed(addNonNegativeConstraint(
              maybeResult->expr, numSymbols, maybeResult->constraint,
              maybeResult->reference, maybeResult->locals)))
        return std::nullopt;
      return maybeResult;
    }

    // Otherwise, we require a modulo. If the extension goes from `m` to `n`
    // bit, then we must modulo by 2^m.
    unsigned width = operand.getType().getIntOrFloatBitWidth();
    // The modulus 2^m must fit in an int64_t.
    if (width >= 63)
      return std::nullopt;
    int64_t mod = pow<int64_t>(2, width);
    return AffineResult{maybeResult->expr % mod, maybeResult->constraint,
                        maybeResult->reference, std::move(maybeResult->locals)};
  }

  if (auto gep = dyn_cast<LLVM::GEPOp>(def)) {
    auto indices = gep.getIndices();
    Type elementType = gep.getElemType();
    AffineExpr expr = getAffineConstantExpr(0, ctx);
    IntegerRelation constraint(space);
    IntegerRelation reference(space);
    SmallVector<LocalDef> locals;
    // We constraint on the sum only.
    DataLayout layout(cast<ModuleOp>(getOperation()));
    for (auto index : indices) {
      if (auto v = dyn_cast<Value>(index)) {
        auto maybe = constructAffineMap(v, dimIndex);
        if (!maybe)
          return std::nullopt;

        expr = expr + maybe->expr * layout.getTypeSize(elementType);
        constraint = constraint.intersect(maybe->constraint);
        reference = reference.intersect(maybe->reference);
        locals = mergeLocalDefs(std::move(locals), maybe->locals, numSymbols);

        if (isa<LLVM::LLVMStructType>(elementType))
          return std::nullopt;
        elementType =
            TypeSwitch<Type, Type>(elementType)
                .Case([](LLVM::LLVMArrayType t) { return t.getElementType(); })
                .Default([](Type t) { return t; });
        // Create the constraint suhc that the entire sum should not overflow.
        if (failed(populateConstraint(expr, numSymbols, v.getType(), constraint,
                                      locals)))
          return std::nullopt;
        syncLocalDefs(reference, locals);
      }
      // TODO: IntegerAttr case
    }
    // GEP always treats its offsets as signed.
    return AffineResult{expr, constraint, reference, std::move(locals)};
  }
  return std::nullopt;
}

ArithToAffinePass::AffineMapResult
ArithToAffinePass::tidyResult(const AffineResult &result, Value value,
                              ValueRange operands) {
  const IntegerRelation &rConstraint = result.constraint;

  // No need to tidy expressions without operands.
  if (operands.empty())
    return {operands, result.expr, rConstraint, result.locals};

  // Compute unused indices.
  unsigned numOperands = operands.size();
  MLIRContext *ctx = operands[0].getContext();
  DenseSet<unsigned> used;
  result.expr.walk([&](AffineExpr expr) {
    if (auto symbol = dyn_cast<AffineSymbolExpr>(expr))
      used.insert(symbol.getPosition());
  });

  // Maps old indices to ones after removal of unused indices, and
  // reconstruct an expression.
  SmallVector<AffineExpr> replacement(numOperands);
  DenseMap<unsigned, unsigned> indexMap;
  for (unsigned i = 0, index = 0; i < numOperands; i++) {
    if (used.contains(i)) {
      replacement[i] = getAffineSymbolExpr(index, ctx);
      indexMap[i] = index;
      index++;
    } else
      replacement[i] = getAffineConstantExpr(0, ctx);
  }
  auto expr = result.expr.replaceSymbols(replacement);

  // Remove columns of constraints, keeping the local variables.
  unsigned numLeft = used.size();
  unsigned numLocals = rConstraint.getNumLocalVars();
  IntegerRelation constraint(
      PresburgerSpace::getSetSpace(numLeft, 0, numLocals));
  // The constraints recorded from nsw/nuw flags, remapped the same way.
  IntegerRelation wrapFlags(
      PresburgerSpace::getSetSpace(numLeft, 0, numLocals));

  // Removes the redundant columns in `from` according to `indexMap` above,
  // and populates `to`.
  const auto removeColumn = [&](const IntegerRelation &from,
                                IntegerRelation &to, bool isEq) {
    unsigned size = isEq ? from.getNumEqualities() : from.getNumInequalities();
    for (unsigned i = 0; i < size; i++) {
      auto row = isEq ? from.getEquality(i) : from.getInequality(i);
      IntVector x(numLeft + numLocals + 1);
      for (auto [before, after] : indexMap)
        x[after] = row[before];
      for (unsigned k = 0; k < numLocals; k++)
        x[numLeft + k] = row[numOperands + k];
      x.back() = row.back();
      if (isEq)
        to.addEquality(x);
      else
        to.addInequality(x);
    }
  };
  removeColumn(rConstraint, constraint, true);
  removeColumn(rConstraint, constraint, false);
  removeColumn(result.reference, wrapFlags, true);
  removeColumn(result.reference, wrapFlags, false);

  // Remap the dividends of the local variable definitions likewise.
  SmallVector<LocalDef> locals;
  locals.reserve(result.locals.size());
  for (const LocalDef &def : result.locals) {
    IntVector nom(numLeft + (def.nom.size() - numOperands));
    for (auto [before, after] : indexMap)
      nom[after] = def.nom[before];
    for (unsigned k = numOperands; k + 1 < def.nom.size(); k++)
      nom[numLeft + (k - numOperands)] = def.nom[k];
    nom.back() = def.nom.back();
    locals.push_back({std::move(nom), def.den});
  }

  OpBuilder builder(ctx);
  auto loc = operands[0].getLoc();
  SmallVector<Value> symbols;
  // For every related operand, cast them to index if they aren't already.
  // We could potentially deduplicate these casts, but it's also possible to
  // leave them for CSE.
  for (auto [i, v] : llvm::enumerate(operands)) {
    if (!used.contains(i))
      continue;

    if (!isa<IndexType>(v.getType())) {
      builder.setInsertionPointAfterValue(v);
      auto cast =
          arith::IndexCastOp::create(builder, loc, builder.getIndexType(), v);
      symbols.push_back(cast.getResult());
    } else {
      symbols.push_back(v);
    }
  }

  // Remove rows of constraints that are already implied by facts known about
  // the operands: their analyzed integer ranges, and bounds obtained via
  // ValueBoundsOpInterface (e.g. an scf.for induction variable is bounded by
  // the loop bounds).
  IntegerRelation reference(PresburgerSpace::getSetSpace(numLeft));
  DenseMap<Value, unsigned> valueToSymbol;
  for (auto [before, after] : indexMap)
    valueToSymbol.try_emplace(operands[before], after);
  for (unsigned i = 0; i < numOperands; i++) {
    auto it = indexMap.find(i);
    if (it == indexMap.end())
      continue;
    addAnalyzedRangeConstraints(operands[i], it->second, reference);
    addValueBoundsConstraints(operands[i], it->second, valueToSymbol,
                              reference);
  }
  // Conditions of `scf.if` ops enclosing all uses of `value` hold wherever
  // the runtime check is emitted, so they are facts as well.
  addBranchConditionConstraints(value, valueToSymbol, reference);
  // The reference relation knows nothing about the local variables so far;
  // intersecting with the flag facts gives them their definitions. Rows
  // involving locals are then only removable when implied by the remaining
  // rows, which keeps the synthesized check sound.
  reference = reference.intersect(wrapFlags);
  constraint.removeRedundantConstraintsWhen(reference);

  return AffineMapResult{symbols, expr, constraint, std::move(locals)};
}

// Adds `min <= var[pos] <= max` rows to `rel`.
void addConstantBounds(IntegerRelation &rel, unsigned pos,
                       const DynamicAPInt &min, const DynamicAPInt &max) {
  SmallVector<DynamicAPInt> lower(rel.getNumVars() + 1);
  lower[pos] = DynamicAPInt(1);
  lower.back() = -min;
  rel.addInequality(lower);
  SmallVector<DynamicAPInt> upper(rel.getNumVars() + 1);
  upper[pos] = DynamicAPInt(-1);
  upper.back() = max;
  rel.addInequality(upper);
}

void ArithToAffinePass::addAnalyzedRangeConstraints(
    Value value, unsigned pos, IntegerRelation &rel) const {
  auto *lattice =
      solver->lookupState<dataflow::IntegerValueRangeLattice>(value);
  if (!lattice || lattice->getValue().isUninitialized())
    return;

  // The symbol is the sign-extended index-cast of `value`, so it is bounded
  // by the signed range of `value`.
  const ConstantIntRanges &range = lattice->getValue().getValue();
  addConstantBounds(rel, pos, DynamicAPInt(range.smin()),
                    DynamicAPInt(range.smax()));
}

// Returns the operation that a ValueBoundsOpInterface query for `value`
// should be directed to: the defining operation for an op result, or the
// operation owning the region for a block argument.
Operation *getOwner(Value value) {
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    Region *region = arg.getOwner()->getParent();
    return region ? region->getParentOp() : nullptr;
  }
  return value.getDefiningOp();
}

void ArithToAffinePass::addValueBoundsConstraints(
    Value value, unsigned pos, const DenseMap<Value, unsigned> &valueToSymbol,
    IntegerRelation &reference) const {
  // Bounds can only be computed for index/integer-typed values, and require
  // the owner of the value to implement ValueBoundsOpInterface (e.g. an
  // scf.for loop for its induction variable).
  if (!value.getType().isIntOrIndex())
    return;
  Operation *owner = getOwner(value);
  if (!owner || !isa<ValueBoundsOpInterface>(owner))
    return;

  MLIRContext *ctx = value.getContext();
  ValueBoundsOptions options;
  options.allowIntegerType = true;

  // Stop the backward traversal at values that the bound should be expressed
  // in terms of: lifted operands (which have symbols in `reference`) and
  // values that cannot be analyzed further. Never stop at the queried value
  // itself.
  auto stopCondition = [&](Value v, std::optional<int64_t> dim,
                           ValueBoundsConstraintSet &) {
    if (v == value)
      return false;
    if (valueToSymbol.contains(v))
      return true;
    Operation *owner = getOwner(v);
    return !owner || !isa<ValueBoundsOpInterface>(owner);
  };

  unsigned numSyms = reference.getNumVars();
  // Constraints derived from the computed bounds, over the reference symbols
  // plus local variables for bound operands without a corresponding symbol.
  IntegerRelation fragment(PresburgerSpace::getSetSpace(numSyms));
  // Maps bound operands (value/dim pairs) to their local variable position
  // in `fragment`.
  SmallVector<std::pair<std::pair<Value, std::optional<int64_t>>, unsigned>>
      locals;

  ValueBoundsConstraintSet::Variable var(value);
  for (BoundType type : {BoundType::LB, BoundType::UB}) {
    AffineMap boundMap;
    ValueDimList mapOperands;
    if (failed(ValueBoundsConstraintSet::computeBound(
            boundMap, mapOperands, type, var, stopCondition, options)))
      continue;

    // Convert the dimensions of the map to symbols: both dimensions and
    // symbols correspond to entries of `mapOperands`.
    unsigned numMapOperands = boundMap.getNumDims() + boundMap.getNumSymbols();
    SmallVector<AffineExpr> dimReplacements, symReplacements;
    for (unsigned i = 0; i < boundMap.getNumDims(); i++)
      dimReplacements.push_back(getAffineSymbolExpr(i, ctx));
    for (unsigned i = 0; i < boundMap.getNumSymbols(); i++)
      symReplacements.push_back(
          getAffineSymbolExpr(boundMap.getNumDims() + i, ctx));
    AffineExpr boundExpr = boundMap.getResult(0).replaceDimsAndSymbols(
        dimReplacements, symReplacements);

    // The computed bound is not necessarily affine (e.g. it may contain a
    // floordiv with a non-constant divisor); skip such bounds.
    std::optional<IntVector> maybeCoeffs =
        extractCoefficients(boundExpr, numMapOperands);
    if (!maybeCoeffs)
      continue;
    const IntVector &coeffs = *maybeCoeffs;

    // Assign a column to each bound operand: the symbol of the corresponding
    // lifted operand if there is one, or a fresh local variable otherwise.
    SmallVector<unsigned> columns(numMapOperands);
    for (unsigned k = 0; k < numMapOperands; k++) {
      if (coeffs[k] == 0)
        continue;
      const std::pair<Value, std::optional<int64_t>> &valueDim = mapOperands[k];
      if (!valueDim.second) {
        auto it = valueToSymbol.find(valueDim.first);
        if (it != valueToSymbol.end()) {
          columns[k] = it->second;
          continue;
        }
      }
      auto *it = llvm::find_if(
          locals, [&](const auto &entry) { return entry.first == valueDim; });
      if (it == locals.end()) {
        unsigned localPos = numSyms + fragment.getNumLocalVars();
        fragment.appendVar(VarKind::Local);
        // Constrain the local variable with the analyzed range of the value.
        if (!valueDim.second)
          addAnalyzedRangeConstraints(valueDim.first, localPos, fragment);
        it = std::prev(locals.insert(locals.end(), {valueDim, localPos}));
      }
      columns[k] = it->second;
    }

    // Lower bound: `symbol - bound >= 0`.
    // Upper bound: `bound - symbol >= 0`.
    DynamicAPInt sign(type == BoundType::LB ? 1 : -1);
    SmallVector<DynamicAPInt> row(fragment.getNumVars() + 1);
    row[pos] = sign;
    for (unsigned k = 0; k < numMapOperands; k++) {
      if (coeffs[k] == 0)
        continue;
      row[columns[k]] -= coeffs[k] * sign;
    }
    row.back() -= coeffs.back() * sign;
    fragment.addInequality(row);
  }

  if (fragment.getNumInequalities() == 0)
    return;

  // Project out the local variables. What remains are constraints over the
  // reference symbols only, which are valid consequences of the computed
  // bounds and the analyzed ranges.
  for (unsigned i = 0, e = fragment.getNumLocalVars(); i < e; i++)
    fragment.projectOut(fragment.getNumDimAndSymbolVars());
  fragment.removeTrivialRedundancy();

  for (unsigned i = 0, e = fragment.getNumInequalities(); i < e; i++)
    reference.addInequality(fragment.getInequality(i));
}

void ArithToAffinePass::addConditionConstraints(
    Value cond, bool negated, const DenseMap<Value, unsigned> &valueToSymbol,
    IntegerRelation &reference) {
  // A then branch gives a conjunction of the `and` operands; an else branch
  // gives the negation, which is a conjunction only for `or` operands.
  if (Operation *def = cond.getDefiningOp()) {
    bool split = negated ? isa<arith::OrIOp>(def) : isa<arith::AndIOp>(def);
    if (split) {
      addConditionConstraints(def->getOperand(0), negated, valueToSymbol,
                              reference);
      addConditionConstraints(def->getOperand(1), negated, valueToSymbol,
                              reference);
      return;
    }
  }

  auto cmp = cond.getDefiningOp<arith::CmpIOp>();
  if (!cmp)
    return;

  arith::CmpIPredicate predicate = cmp.getPredicate();
  if (negated)
    predicate = arith::invertPredicate(predicate);

  unsigned numSymbols = reference.getNumVars();
  // The difference `lhs - rhs` as a row over the reference symbols.
  SmallVector<Value> leaves;
  collectOperandsUpChain(cmp.getLhs(), leaves);
  collectOperandsUpChain(cmp.getRhs(), leaves);
  SmallVector<Value> uniq;
  DenseMap<Value, unsigned> localIndex;
  for (Value leaf : leaves)
    if (localIndex.try_emplace(leaf, uniq.size()).second)
      uniq.push_back(leaf);
  DimIndex dimIndex;
  for (auto [i, v] : llvm::enumerate(uniq))
    dimIndex[v] = i;

  const auto sideToRow = [&](Value side) -> std::optional<IntVector> {
    auto result = constructAffineMap(side, dimIndex);
    if (!result)
      return std::nullopt;
    auto coeffs = extractCoefficients(result->expr, uniq.size());
    if (!coeffs)
      return std::nullopt;
    IntVector row(numSymbols + 1);
    for (unsigned i = 0; i + 1 < coeffs->size(); i++) {
      if ((*coeffs)[i] == 0)
        continue;
      auto it = valueToSymbol.find(uniq[i]);
      if (it == valueToSymbol.end())
        return std::nullopt;
      row[it->second] = (*coeffs)[i];
    }
    row.back() = coeffs->back();
    return row;
  };

  auto lhs = sideToRow(cmp.getLhs());
  auto rhs = sideToRow(cmp.getRhs());
  if (!lhs || !rhs)
    return;
  IntVector diff(numSymbols + 1);
  for (unsigned i = 0; i <= numSymbols; i++)
    diff[i] = (*lhs)[i] - (*rhs)[i];

  // The symbols are the sign-extended lifted operands, so only signed
  // comparisons (and equality) are meaningful here.
  switch (predicate) {
  case arith::CmpIPredicate::eq:
    reference.addEquality(diff);
    break;
  case arith::CmpIPredicate::sge:
    reference.addInequality(diff);
    break;
  case arith::CmpIPredicate::sgt:
    diff.back() -= 1;
    reference.addInequality(diff);
    break;
  case arith::CmpIPredicate::sle:
    for (auto &e : diff)
      e *= -1;
    reference.addInequality(diff);
    break;
  case arith::CmpIPredicate::slt:
    for (auto &e : diff)
      e *= -1;
    diff.back() -= 1;
    reference.addInequality(diff);
    break;
  default:
    break;
  }
}

void ArithToAffinePass::addBranchConditionConstraints(
    Value value, const DenseMap<Value, unsigned> &valueToSymbol,
    IntegerRelation &reference) {
  // Find the least common ancestor region of all uses of `value`; the
  // conditions of the `scf.if` ops enclosing it hold at every use.
  Region *lca = nullptr;
  for (OpOperand &use : value.getUses()) {
    Region *region = use.getOwner()->getBlock()->getParent();
    while (lca && lca != region && !lca->isAncestor(region)) {
      Operation *op = lca->getParentOp();
      lca = op ? op->getParentRegion() : nullptr;
    }
    if (!lca)
      lca = region;
  }
  if (!lca)
    return;

  // Walk up the region tree. `child` is always the region directly inside
  // `op` that lies on the path from `lca`.
  for (Region *child = lca; Operation *op = child->getParentOp();) {
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      bool inElse = child == &ifOp.getElseRegion();
      addConditionConstraints(ifOp.getCondition(), /*negated=*/inElse,
                              valueToSymbol, reference);
    }
    child = op->getParentRegion();
    if (!child)
      break;
  }
}

std::optional<ArithToAffinePass::AffineMapResult>
ArithToAffinePass::valueToAffine(Value value) {
  // We trace back the arithmetic computations of the given value till a valid
  // symbol. If eventually we failed to hit anything convertible to symbol, we
  // abort.
  SmallVector<Value> operands;
  collectOperandsUpChain(value, operands);

  // Now compute an affine map and the bounds of the symbols when raising to
  // index. We must guarantee that the index type does not overflow. The arith
  // operations are not necessarily affine, so we do not mutate IR here.
  DimIndex dimIndex;
  for (auto [i, v] : llvm::enumerate(operands))
    dimIndex[v] = i;

  auto maybeResult = constructAffineMap(value, dimIndex);
  if (!maybeResult)
    return std::nullopt;

  return tidyResult(*maybeResult, value, operands);
}

ArithToAffinePass::LiftResult ArithToAffinePass::lift(Value value) {
  auto liftResult = valueToAffine(value);
  // Give up lifting if we cannot make it affine.
  if (!liftResult)
    return {Value(), Value()};

  // Construct an `affine.apply` with the given symbols and affine map.
  MLIRContext *ctx = value.getContext();
  OpBuilder builder(ctx);
  builder.setInsertionPointAfterValue(value);
  auto loc = value.getLoc();
  auto affineMap =
      AffineMap::get(0, liftResult->operands.size(), liftResult->expr);
  Value apply = affine::AffineApplyOp::create(
      builder, loc, builder.getIndexType(), affineMap, liftResult->operands);
  SmallVector<Value> checkValues;
  SmallVector<Bound> checkBounds;
  for (Value operand : liftResult->operands) {
    if (auto cast = operand.getDefiningOp<arith::IndexCastOp>()) {
      checkValues.push_back(cast.getIn());
      checkBounds.push_back(getSignedBounds(cast.getIn().getType()));
    } else {
      checkValues.push_back(operand);
      checkBounds.push_back(bounds(operand));
    }
  }
  Value condition =
      synthesizeCheck(builder, loc, liftResult->constraint, liftResult->locals,
                      checkValues, checkBounds);

  return {apply, condition};
}

Bound ArithToAffinePass::bounds(Value value) const {
  if (auto *lattice =
          solver->lookupState<dataflow::IntegerValueRangeLattice>(value);
      lattice && !lattice->getValue().isUninitialized()) {
    const ConstantIntRanges &range = lattice->getValue().getValue();
    return {DynamicAPInt(range.smin()), DynamicAPInt(range.smax())};
  }
  if (value.getType().isInteger())
    return getSignedBounds(value.getType());
  auto power = pow(DynamicAPInt(2), 63);
  return {-power, power - 1};
}

Value findGEPBase(Value addr) {
  if (isa<BlockArgument>(addr))
    return addr;

  if (auto gep = dyn_cast<LLVM::GEPOp>(addr.getDefiningOp()))
    return findGEPBase(gep.getBase());

  return addr;
}

std::optional<IntVector>
ArithToAffinePass::computeOuterBound(Value inner, BoundType boundType,
                                     Operation *loop,
                                     SmallVectorImpl<Value> &hoistedOperands,
                                     DenseMap<Value, unsigned> &hoistedIndex) {
  if (!inner.getType().isIntOrIndex())
    return std::nullopt;
  Operation *owner = getOwner(inner);
  if (!owner || !isa<ValueBoundsOpInterface>(owner))
    return std::nullopt;

  MLIRContext *ctx = inner.getContext();
  ValueBoundsOptions options;
  options.allowIntegerType = true;

  // Stop the backward traversal at values defined outside the loop: the
  // bound must be expressed in terms of those so that it is available before
  // the loop. Also stop at values that cannot be analyzed further.
  auto stopCondition = [&](Value v, std::optional<int64_t> dim,
                           ValueBoundsConstraintSet &) {
    if (v == inner)
      return false;
    if (!valueInLoop(v, loop))
      return true;
    Operation *owner = getOwner(v);
    return !owner || !isa<ValueBoundsOpInterface>(owner);
  };

  AffineMap boundMap;
  ValueDimList mapOperands;
  if (failed(ValueBoundsConstraintSet::computeBound(
          boundMap, mapOperands, boundType,
          ValueBoundsConstraintSet::Variable(inner), stopCondition, options)))
    return std::nullopt;

  // Convert the dimensions of the map to symbols: both dimensions and
  // symbols correspond to entries of `mapOperands`.
  unsigned numMapOperands = boundMap.getNumDims() + boundMap.getNumSymbols();
  SmallVector<AffineExpr> dimReplacements, symReplacements;
  for (unsigned i = 0; i < boundMap.getNumDims(); i++)
    dimReplacements.push_back(getAffineSymbolExpr(i, ctx));
  for (unsigned i = 0; i < boundMap.getNumSymbols(); i++)
    symReplacements.push_back(
        getAffineSymbolExpr(boundMap.getNumDims() + i, ctx));
  AffineExpr boundExpr = boundMap.getResult(0).replaceDimsAndSymbols(
      dimReplacements, symReplacements);

  // The bound is not necessarily affine (e.g. it may contain a floordiv with
  // a non-constant divisor); reject such bounds.
  std::optional<IntVector> maybeCoeffs =
      extractCoefficients(boundExpr, numMapOperands);
  if (!maybeCoeffs)
    return std::nullopt;

  // Register the bound operands as hoisted symbols. They must all be defined
  // outside the loop for the bound to be available there.
  SmallVector<std::pair<unsigned, DynamicAPInt>> terms;
  for (unsigned k = 0; k < numMapOperands; k++) {
    if ((*maybeCoeffs)[k] == 0)
      continue;
    if (mapOperands[k].second)
      return std::nullopt;
    Value atom = mapOperands[k].first;
    if (valueInLoop(atom, loop))
      return std::nullopt;
    auto [it, inserted] =
        hoistedIndex.try_emplace(atom, hoistedOperands.size());
    if (inserted)
      hoistedOperands.push_back(atom);
    terms.push_back({it->second, (*maybeCoeffs)[k]});
  }

  IntVector result(hoistedOperands.size() + 1);
  for (const auto &[index, coeff] : terms)
    result[index] += coeff;
  result.back() = maybeCoeffs->back();
  return result;
}

void ArithToAffinePass::liftLoadAddress(LLVM::LoadOp load) {
  OpBuilder builder(load);
  Value addr = load.getAddr();
  Location loc = load.getLoc();
  LiftResult result = lift(addr);
  if (!result.apply)
    return;
  auto branch = scf::IfOp::create(
      builder, loc, result.condition,
      [&](OpBuilder &builder, Location loc) {
        Value gepBase = findGEPBase(addr);
        auto cast = arith::IndexCastOp::create(
            builder, loc, builder.getI64Type(), result.apply);
        auto gep = LLVM::GEPOp::create(builder, loc, addr.getType(),
                                       builder.getI8Type(), gepBase, {cast});
        scf::YieldOp::create(builder, loc, gep.getResult());
      },
      [&](OpBuilder &builder, Location loc) {
        scf::YieldOp::create(builder, loc, addr);
      });
  load.setOperand(branch.getResult(0));
}

void ArithToAffinePass::liftLoop(Operation *loop) {
  // Collect the loads directly inside `loop`; loads of nested loops are
  // handled when those loops are processed.
  SmallVector<LLVM::LoadOp> loads;
  loop->walk([&](LLVM::LoadOp load) {
    if (load->getParentOfType<LoopLikeOpInterface>().getOperation() == loop)
      loads.push_back(load);
  });
  if (loads.empty())
    return;

  const auto fallback = [&]() {
    for (auto load : loads)
      liftLoadAddress(load);
  };

  // Union of the leaf operands of all the load address expressions.
  SmallVector<Value> operands;
  DenseSet<Value> seen;
  for (auto load : loads) {
    SmallVector<Value> leaves;
    collectOperandsUpChain(load.getAddr(), leaves);
    for (Value v : leaves)
      if (seen.insert(v).second)
        operands.push_back(v);
  }
  unsigned numSymbols = operands.size();

  DimIndex dimIndex;
  for (auto [i, v] : llvm::enumerate(operands))
    dimIndex[v] = i;

  // Symbols whose value is defined inside the loop (e.g. the induction
  // variable) cannot be referenced by a check emitted before the loop.
  SmallVector<bool> inner(numSymbols);
  for (unsigned j = 0; j < numSymbols; j++)
    inner[j] = valueInLoop(operands[j], loop);

  // Symbol table of the hoisted check: the outer operands first, in order.
  SmallVector<Value> hOperands;
  DenseMap<Value, unsigned> hIndex;
  SmallVector<unsigned> allToHoisted(numSymbols);
  for (unsigned j = 0; j < numSymbols; j++) {
    if (inner[j])
      continue;
    allToHoisted[j] = hOperands.size();
    hIndex[operands[j]] = hOperands.size();
    hOperands.push_back(operands[j]);
  }

  // Intersect the constraints of all the loads.
  PresburgerSpace space = PresburgerSpace::getSetSpace(numSymbols);
  IntegerRelation constraint(space);
  IntegerRelation reference(space);
  SmallVector<LocalDef> locals;
  SmallVector<std::optional<AffineExpr>> exprs;
  bool any = false;
  for (auto load : loads) {
    auto result = constructAffineMap(load.getAddr(), dimIndex);
    if (!result) {
      exprs.push_back(std::nullopt);
      continue;
    }
    // Bring the local variables of `result` into the combined space.
    unsigned offset = locals.size();
    for (const LocalDef &def : result->locals) {
      IntVector nom = shift(def.nom, numSymbols, offset);
      constraint.addLocalFloorDiv(nom, def.den);
      reference.addLocalFloorDiv(nom, def.den);
      locals.push_back({nom, def.den});
    }
    constraint = constraint.intersect(result->constraint);
    reference = reference.intersect(result->reference);
    exprs.push_back(result->expr);
    any = true;
  }
  if (!any)
    return fallback();

  // Record facts known about the symbols (analyzed ranges, value bounds) in
  // a local-free relation, then merge them into `reference`.
  IntegerRelation facts(PresburgerSpace::getSetSpace(numSymbols));
  DenseMap<Value, unsigned> valueToSymbol;
  for (unsigned j = 0; j < numSymbols; j++)
    valueToSymbol[operands[j]] = j;
  for (unsigned j = 0; j < numSymbols; j++) {
    addAnalyzedRangeConstraints(operands[j], j, facts);
    addValueBoundsConstraints(operands[j], j, valueToSymbol, facts);
  }
  unsigned numLocals = locals.size();
  for (unsigned i = 0, e = facts.getNumInequalities(); i < e; i++) {
    auto row = facts.getInequality(i);
    IntVector padded(numSymbols + numLocals + 1);
    for (unsigned j = 0; j < numSymbols; j++)
      padded[j] = row[j];
    padded.back() = row.back();
    reference.addInequality(padded);
  }
  constraint.removeRedundantConstraintsWhen(reference);

  // The definitions of local variables must be materializable before the
  // loop, so they must not involve inner symbols.
  for (const LocalDef &def : locals)
    for (unsigned j = 0; j < numSymbols; j++)
      if (inner[j] && def.nom[j] != 0)
        return fallback();

  // Determine the inner symbols the remaining constraints refer to, and
  // compute their bounds in terms of the outer symbols.
  SmallVector<bool> needed(numSymbols, false);
  const auto scan = [&](bool isEq) {
    unsigned numRows =
        isEq ? constraint.getNumEqualities() : constraint.getNumInequalities();
    for (unsigned i = 0; i < numRows; i++) {
      auto row = isEq ? constraint.getEquality(i) : constraint.getInequality(i);
      for (unsigned j = 0; j < numSymbols; j++)
        if (inner[j] && row[j] != 0)
          needed[j] = true;
    }
  };
  scan(true);
  scan(false);

  SmallVector<std::optional<IntVector>> lower(numSymbols), upper(numSymbols);
  for (unsigned j = 0; j < numSymbols; j++) {
    if (!needed[j])
      continue;
    lower[j] =
        computeOuterBound(operands[j], BoundType::LB, loop, hOperands, hIndex);
    upper[j] =
        computeOuterBound(operands[j], BoundType::UB, loop, hOperands, hIndex);
    if (!lower[j] || !upper[j])
      return fallback();
  }

  // Substitute the worst-case bound of every inner symbol into the
  // constraints, yielding a relation over the outer symbols only.
  unsigned numOuter = hOperands.size();
  IntegerRelation hoisted(PresburgerSpace::getSetSpace(numOuter, 0, numLocals));
  IntegerRelation hReference(PresburgerSpace::getSetSpace(numOuter));
  for (unsigned j = 0; j < numOuter; j++)
    addAnalyzedRangeConstraints(hOperands[j], j, hReference);
  SmallVector<LocalDef> hLocals;
  for (unsigned i = 0; i < numLocals; i++) {
    const LocalDef &def = locals[i];
    IntVector nom(numOuter + i + 1);
    for (unsigned j = 0; j < numSymbols; j++)
      if (!inner[j])
        nom[allToHoisted[j]] = def.nom[j];
    for (unsigned k = 0; k < i; k++)
      nom[numOuter + k] = def.nom[numSymbols + k];
    nom.back() = def.nom.back();
    hLocals.push_back({nom, def.den});
    hoisted.addLocalFloorDiv(nom, def.den);
    hReference.addLocalFloorDiv(nom, def.den);
  }
  const auto substitute = [&](bool isEq) {
    unsigned numRows =
        isEq ? constraint.getNumEqualities() : constraint.getNumInequalities();
    for (unsigned i = 0; i < numRows; i++) {
      auto row = isEq ? constraint.getEquality(i) : constraint.getInequality(i);
      IntVector hoistedRow(numOuter + numLocals + 1);
      hoistedRow.back() = row.back();
      for (unsigned j = 0; j < numSymbols; j++) {
        if (!inner[j]) {
          hoistedRow[allToHoisted[j]] = row[j];
          continue;
        }
        const DynamicAPInt &coeff = row[j];
        if (coeff == 0)
          continue;
        if (isEq)
          return false;
        // The row must hold for every value of the inner symbol, so
        // substitute the worst-case bound.
        const IntVector &bound = coeff > 0 ? *lower[j] : *upper[j];
        for (unsigned t = 0; t + 1 < bound.size(); t++)
          hoistedRow[t] += coeff * bound[t];
        hoistedRow.back() += coeff * bound.back();
      }
      for (unsigned k = 0; k < numLocals; k++)
        hoistedRow[numOuter + k] = row[numSymbols + k];
      if (isEq)
        hoisted.addEquality(hoistedRow);
      else
        hoisted.addInequality(hoistedRow);
    }
    return true;
  };
  if (!substitute(true) || !substitute(false))
    return fallback();

  hoisted.removeRedundantConstraintsWhen(hReference);
  hoisted.removeTrivialRedundancy();

  Location loc = loop->getLoc();
  OpBuilder builder(loop);

  // Cast the outer operands to index for the affine.apply's.
  SmallVector<Value> outerIndex(numSymbols);
  for (unsigned j = 0; j < numSymbols; j++) {
    if (inner[j] || !operands[j].getType().isIntOrIndex())
      continue;
    Value v = operands[j];
    if (!isa<IndexType>(v.getType()))
      v = arith::IndexCastOp::create(builder, loc, builder.getIndexType(), v);
    outerIndex[j] = v;
  }

  // Replaces the address of `load` with a fresh GEP based on the lifted
  // affine expression. `mapping` maps the values of the original loop to
  // those of its clone, if any.
  const auto applyLift = [&](LLVM::LoadOp load, AffineExpr expr,
                             IRMapping *mapping) {
    OpBuilder b(load);
    MLIRContext *ctx = load.getContext();
    // Drop the symbols that do not appear in the expression.
    SmallVector<int> remap(numSymbols, -1);
    SmallVector<unsigned> used;
    expr.walk([&](AffineExpr e) {
      if (auto symbol = dyn_cast<AffineSymbolExpr>(e)) {
        unsigned pos = symbol.getPosition();
        if (remap[pos] == -1) {
          remap[pos] = used.size();
          used.push_back(pos);
        }
      }
    });
    SmallVector<AffineExpr> replacement(numSymbols,
                                        getAffineConstantExpr(0, ctx));
    SmallVector<Value> applyOperands;
    for (unsigned j : used) {
      replacement[j] = getAffineSymbolExpr(remap[j], ctx);
      Value v;
      if (!inner[j]) {
        v = outerIndex[j];
      } else {
        v = operands[j];
        if (mapping)
          v = mapping->lookupOrDefault(v);
        if (!isa<IndexType>(v.getType()))
          v = arith::IndexCastOp::create(b, loc, b.getIndexType(), v);
      }
      applyOperands.push_back(v);
    }
    auto map = AffineMap::get(0, used.size(), expr.replaceSymbols(replacement));
    Value apply = affine::AffineApplyOp::create(b, loc, b.getIndexType(), map,
                                                applyOperands);
    Value addr = load.getAddr();
    Value base = findGEPBase(addr);
    auto offset = arith::IndexCastOp::create(b, loc, b.getI64Type(), apply);
    Value gep = LLVM::GEPOp::create(b, loc, addr.getType(), b.getI8Type(), base,
                                    {offset});
    load.setOperand(gep);
  };

  // If nothing remains to be checked, lift in place.
  if (hoisted.getNumInequalities() == 0 && hoisted.getNumEqualities() == 0) {
    for (unsigned i = 0; i < loads.size(); i++)
      if (exprs[i])
        applyLift(loads[i], *exprs[i], nullptr);
    return;
  }

  // Synthesize the check before the loop.
  SmallVector<Value> checkValues;
  SmallVector<Bound> checkBounds;
  for (Value v : hOperands) {
    if (auto cast = v.getDefiningOp<arith::IndexCastOp>()) {
      checkValues.push_back(cast.getIn());
      checkBounds.push_back(getSignedBounds(cast.getIn().getType()));
    } else {
      checkValues.push_back(v);
      checkBounds.push_back(bounds(v));
    }
  }
  Value condition =
      synthesizeCheck(builder, loc, hoisted, hLocals, checkValues, checkBounds);

  // Emit `if (check) { lifted loop } else { original loop }`.
  auto ifOp = scf::IfOp::create(builder, loc, loop->getResultTypes(), condition,
                                /*withElseRegion=*/true);

  // Then block: clone the loop, apply lifts, and yield the results.
  {
    Block &thenBlock = ifOp.getThenRegion().front();
    if (thenBlock.mightHaveTerminator())
      thenBlock.getTerminator()->erase();
    OpBuilder b(&thenBlock, thenBlock.end());
    IRMapping mapping;
    Operation *cloned = b.clone(*loop, mapping);
    for (unsigned i = 0; i < loads.size(); i++)
      if (exprs[i])
        applyLift(cast<LLVM::LoadOp>(mapping.lookup(loads[i].getOperation())),
                  *exprs[i], &mapping);
    scf::YieldOp::create(b, loc, cloned->getResults());
  }

  // Else block: move the original loop and yield its results.
  {
    Block &elseBlock = ifOp.getElseRegion().front();
    if (elseBlock.mightHaveTerminator())
      elseBlock.getTerminator()->erase();
    loop->moveBefore(&elseBlock, elseBlock.end());
    OpBuilder b(&elseBlock, elseBlock.end());
    scf::YieldOp::create(b, loc, loop->getResults());
  }

  // Uses of the loop results outside the `if` now refer to its results.
  for (auto [result, ifResult] :
       llvm::zip_equal(loop->getResults(), ifOp.getResults()))
    result.replaceUsesWithIf(ifResult, [&](OpOperand &use) {
      return !ifOp->isAncestor(use.getOwner());
    });
}

void ArithToAffinePass::runOnOperation() {
  mlir::DataFlowSolver mySolver;
  solver = &mySolver;
  Operation *module = getOperation();

  dataflow::loadBaselineAnalyses(*solver);
  solver->load<mlir::dataflow::IntegerRangeAnalysis>();
  if (failed(solver->initializeAndRun(module)))
    // TODO: This should still work but less accurate.
    return;

  // For every loop, intersect the constraints of the loads in its region and
  // emit a single check before it. Process innermost loops first, so that
  // each load is handled by its closest enclosing loop.
  SmallVector<Operation *> loops;
  module->walk(
      [&](LoopLikeOpInterface op) { loops.push_back(op.getOperation()); });
  for (Operation *loop : llvm::reverse(loops))
    liftLoop(loop);

  // Loads outside any loop are lifted individually.
  module->walk([&](LLVM::LoadOp load) {
    if (load->getParentOfType<LoopLikeOpInterface>())
      return;
    liftLoadAddress(load);
  });
}

} // namespace
