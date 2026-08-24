#include "mlir/Conversion/ArithToAffine/ArithToAffine.h"

#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/PresburgerSpace.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemoryAccessOpInterfaces.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Pass/Pass.h"
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
    // Definitions of the local variables of `constraint`, in column order.
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
  AffineMapResult tidyResult(const AffineResult &result, ValueRange operands);
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
  void getDependentDialects(DialectRegistry &registry) const override {
    impl::ConvertArithToAffineBase<ArithToAffinePass>::getDependentDialects(
        registry);
    scf::registerValueBoundsOpInterfaceExternalModels(registry);
  }
  // Computes the offset and represent it as an affine map for `value`.
  LiftResult lift(Value value);
  Value synthesizeCheck(OpBuilder &builder, Location loc,
                        const IntegerRelation &rel, ValueRange operands,
                        ArrayRef<LocalDef> localDefs);
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
  constraint.simplify();
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

Value ArithToAffinePass::synthesizeCheck(OpBuilder &builder, Location loc,
                                         const IntegerRelation &rel,
                                         ValueRange inOperands,
                                         ArrayRef<LocalDef> localDefs) {
  // Compute the minimum and maximum for the operands.
  using Bound = std::pair</*min*/ DynamicAPInt, /*max*/ DynamicAPInt>;
  SmallVector<Bound> bounds;
  // The values of the columns of `rel`: the symbols followed by the local
  // variables materialized from their definitions.
  SmallVector<Value> values;

  bounds.reserve(inOperands.size());
  values.reserve(inOperands.size());
  for (auto operand : inOperands) {
    auto cast = llvm::cast<arith::IndexCastOp>(operand.getDefiningOp());
    auto type = cast.getIn().getType();
    bounds.push_back(getSignedBounds(type));
    values.push_back(cast.getIn());
  }
  assert(rel.getNumLocalVars() == localDefs.size() &&
         "local variable definitions out of sync with constraint");

  unsigned numSymbols = values.size();
  unsigned numLocals = localDefs.size();

  // Returns whether `row` is a positive multiple of `pattern`, i.e., whether
  // the two are logically equivalent as `>= 0` constraints.
  const auto isPositiveMultiple = [](ArrayRef<DynamicAPInt> row,
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
  };
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
        bitWidth = std::max(
            bitWidth,
            1 + (int)cast<IntegerType>(values[j].getType()).getWidth());
    return bitWidth;
  };
  // Emits `coeffs * values + constant` in `intType`.
  const auto emitLinearForm = [&](ArrayRef<DynamicAPInt> coeffs,
                                  const DynamicAPInt &constant, Type intType) {
    Value v =
        arith::ConstantIntOp::create(builder, loc, intType, (int64_t)constant);
    for (unsigned j = 0; j < coeffs.size(); j++) {
      if (coeffs[j] == 0)
        continue;
      auto cst = arith::ConstantIntOp::create(builder, loc, intType,
                                              (int64_t)coeffs[j]);
      auto ext = arith::ExtSIOp::create(builder, loc, intType, values[j]);
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
    ArrayRef<DynamicAPInt> dividend(def.nom);
    auto [min, max] = rangeOf(dividend.drop_back(), def.nom.back());
    // The type must additionally hold the (positive) divisor itself.
    int bitWidth = std::max(widthFor(dividend.drop_back(), min, max),
                            log2(def.den) + 2);
    Type intType = IntegerType::get(builder.getContext(), bitWidth);
    Value v =
        emitLinearForm(dividend.drop_back(), def.nom.back(), intType);
    auto divisor = arith::ConstantIntOp::create(builder, loc, intType,
                                                (int64_t)def.den);
    auto zero = arith::ConstantIntOp::create(builder, loc, intType, 0);
    auto one = arith::ConstantIntOp::create(builder, loc, intType, 1);
    // divsi/remsi truncate towards zero; correct to floor division.
    Value q = arith::DivSIOp::create(builder, loc, v, divisor);
    Value r = arith::RemSIOp::create(builder, loc, v, divisor);
    Value negRem =
        arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt, r, zero);
    q = arith::SelectOp::create(builder, loc, negRem,
                                arith::SubIOp::create(builder, loc, q, one), q);
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
    return AffineResult{expr, universe, {}};
  }

  // We will not trace upwards from block arguments in `valueToAffine()`.
  Operation *def = value.getDefiningOp();
  if (!def) {
    AffineExpr expr = getAffineSymbolExpr(dimIndex.at(value), ctx);
    return AffineResult{expr, universe, {}};
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
    auto locals = mergeLocalDefs(std::move(l->locals), r->locals, numSymbols);
    if (failed(populateConstraint(expr, numSymbols, def->getResult(0).getType(),
                                  constraint, locals)))
      return std::nullopt;

    return AffineResult{expr, constraint, std::move(locals)};
  }

  if (isa<arith::ExtSIOp>(def)) {
    // This never imposes any extra constraint, since we chose our internal
    // representation as signed.
    return constructAffineMap(def->getOperand(0), dimIndex);
  }

  if (auto ext = dyn_cast<arith::ExtUIOp>(def)) {
    // We require a modulo. If the extension goes from `m` to `n` bit, then
    // we must modulo by 2^m.
    auto operand = ext.getIn();
    auto maybeResult = constructAffineMap(operand, dimIndex);
    if (!maybeResult)
      return std::nullopt;

    unsigned width = operand.getType().getIntOrFloatBitWidth();
    // The modulus 2^m must fit in an int64_t.
    if (width >= 63)
      return std::nullopt;
    int64_t mod = pow<int64_t>(2, width);
    return AffineResult{maybeResult->expr % mod,
                        maybeResult->constraint,
                        std::move(maybeResult->locals)};
  }

  if (auto gep = dyn_cast<LLVM::GEPOp>(def)) {
    auto indices = gep.getIndices();
    Type elementType = gep.getElemType();
    AffineExpr expr = getAffineConstantExpr(0, ctx);
    IntegerRelation constraint(space);
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
      }
      // TODO: IntegerAttr case
    }
    // GEP always treats its offsets as signed.
    return AffineResult{expr, constraint, std::move(locals)};
  }
  return std::nullopt;
}

ArithToAffinePass::AffineMapResult
ArithToAffinePass::tidyResult(const AffineResult &result, ValueRange operands) {
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
  const auto removeColumn = [&](bool isEq) {
    unsigned size = isEq ? rConstraint.getNumEqualities()
                         : rConstraint.getNumInequalities();
    for (unsigned i = 0; i < size; i++) {
      auto row =
          isEq ? rConstraint.getEquality(i) : rConstraint.getInequality(i);
      IntVector x(numLeft + numLocals + 1);
      for (auto [before, after] : indexMap)
        x[after] = row[before];
      for (unsigned k = 0; k < numLocals; k++)
        x[numLeft + k] = row[numOperands + k];
      x.back() = row.back();
      if (isEq)
        constraint.addEquality(x);
      else
        constraint.addInequality(x);
    }
  };
  removeColumn(true);
  removeColumn(false);

  // Remap the dividends of the local variable definitions likewise.
  SmallVector<LocalDef> locals;
  locals.reserve(result.locals.size());
  for (const LocalDef &def : result.locals) {
    IntVector dividend(numLeft + (def.nom.size() - numOperands));
    for (auto [before, after] : indexMap)
      dividend[after] = def.nom[before];
    for (unsigned k = numOperands; k + 1 < def.nom.size(); k++)
      dividend[numLeft + (k - numOperands)] = def.nom[k];
    dividend.back() = def.nom.back();
    locals.push_back({std::move(dividend), def.den});
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
  // The reference relation knows nothing about the local variables; append
  // them as unconstrained variables so the spaces are compatible. Rows
  // involving locals are then only removable when implied by the remaining
  // rows, which keeps the synthesized check sound.
  reference.appendVar(VarKind::Local, numLocals);
  llvm::errs() << "constraint:\n";
  constraint.dump();
  llvm::errs() << "\nreference:\n";
  reference.dump();
  constraint.removeRedundantConstraintsWhen(reference);
  llvm::errs() << "constraint simplified:\n";
  constraint.dump();

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

  return tidyResult(*maybeResult, operands);
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
  Value condition =
      synthesizeCheck(builder, loc, liftResult->constraint,
                      liftResult->operands, liftResult->locals);

  return {apply, condition};
}

Value findGEPBase(Value addr) {
  if (isa<BlockArgument>(addr))
    return addr;

  if (auto gep = dyn_cast<LLVM::GEPOp>(addr.getDefiningOp()))
    return findGEPBase(gep.getBase());

  return addr;
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

  // Collect loads and stores and attempt to lift them.
  // For now we consider only indexed load/stores. LLVM loads/stores can
  // wait for later.
  SmallVector<Value> tolift;
  module->walk([&](Operation *op) {
    if (auto indexed = dyn_cast<memref::IndexedAccessOpInterface>(op)) {
      for (auto index : indexed.getIndices())
        tolift.push_back(index);
      return;
    }

    if (auto load = dyn_cast<LLVM::LoadOp>(op)) {
      tolift.push_back(load);
      return;
    }
  });

  for (auto value : tolift) {
    if (auto load = dyn_cast<LLVM::LoadOp>(value.getDefiningOp())) {
      OpBuilder builder(load);
      auto addr = load.getAddr();
      auto loc = load.getLoc();
      if (auto result = lift(addr); result.apply) {
        auto branch = scf::IfOp::create(
            builder, loc, result.condition,
            [&](OpBuilder &builder, Location loc) {
              auto gepBase = findGEPBase(addr);
              auto cast = arith::IndexCastOp::create(
                  builder, loc, builder.getI64Type(), result.apply);
              auto gep =
                  LLVM::GEPOp::create(builder, loc, addr.getType(),
                                      builder.getI8Type(), gepBase, {cast});
              scf::YieldOp::create(builder, loc, gep.getResult());
            },
            [&](OpBuilder &builder, Location loc) {
              scf::YieldOp::create(builder, loc, addr);
            });
        load.setOperand(branch.getResult(0));
      }
    }
  }
}

} // namespace
