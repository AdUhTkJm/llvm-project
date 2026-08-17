#include "mlir/Conversion/ArithToAffine/ArithToAffine.h"

#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/PresburgerSpace.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemoryAccessOpInterfaces.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
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

struct ArithToAffinePass : impl::ConvertArithToAffineBase<ArithToAffinePass> {
  using Base::Base;

  struct AffineMapResult {
    SmallVector<Value> operands;
    AffineExpr expr;
    IntegerRelation constraint;
  };

  struct AffineResult {
    // The affine expression from symbols to the value.
    AffineExpr expr;
    // The constraints on the symbols such that intermediate results will
    // not overflow.
    IntegerRelation constraint;
  };

  struct LiftResult {
    Value apply;
    Value condition;
  };

  // Maps each affine value to its dimension index in a presburger relation.
  using DimIndex = llvm::DenseMap<Value, int>;

  std::optional<AffineMapResult> valueToAffine(Value value);
  std::optional<AffineResult> constructAffineMap(Value value, const DimIndex &dimIndex);
  AffineMapResult tidyResult(const AffineResult &result, ValueRange operands);
  // Computes the offset and represent it as an affine map for `value`.
  LiftResult lift(Value value);
  Value synthesizeCheck(OpBuilder &builder, Location loc, const IntegerRelation &rel, ValueRange operands);
  void runOnOperation() override;
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


using IntVector = SmallVector<llvm::DynamicAPInt>;
// Constructs a coefficient vector for the symbols involved in `expr`.
// The vector is of the same form as rows of IntegerRelation, namely
// the first `numSymbols` elements are coefficients for symbol 0..n-1
// respectively, and the last element is a constant term.
std::optional<IntVector> extractCoefficients(AffineExpr expr, int numSymbols) {
  IntVector result(numSymbols + 1);

  if (auto c = dyn_cast<AffineConstantExpr>(expr)) {
    result.back() += c.getValue();
    return result;
  }
  
  if (auto s = dyn_cast<AffineSymbolExpr>(expr)) {
    result[s.getPosition()] += 1;
    return result;
  }

  if (auto bin = dyn_cast<AffineBinaryOpExpr>(expr)) {
    switch (bin.getKind()) {
    case AffineExprKind::Add: {
      auto l = extractCoefficients(bin.getLHS(), numSymbols);
      auto r = extractCoefficients(bin.getRHS(), numSymbols);
      if (!l || !r)
        return std::nullopt;

      for (int i = 0; i < numSymbols + 1; i++)
        result[i] = (*l)[i] + (*r)[i];

      return result;
    }

    case AffineExprKind::Mul:
      if (auto cst = dyn_cast<AffineConstantExpr>(bin.getLHS())) {
        auto coeffs = extractCoefficients(bin.getRHS(), numSymbols);
        if (!coeffs)
          return std::nullopt;
        for (auto &coeff : *coeffs)
          coeff *= cst.getValue();
        return coeffs;
      }

      if (auto cst = dyn_cast<AffineConstantExpr>(bin.getRHS())) {
        auto coeffs = extractCoefficients(bin.getLHS(), numSymbols);
        if (!coeffs)
          return std::nullopt;
        for (auto &coeff : *coeffs)
          coeff *= cst.getValue();
        return coeffs;
      }
      break;
    
    default:
      ; // fallthrough to the unreachable below.
    }
  }

  return std::nullopt;
}

// Computes x^y for DynamicAPInt.
DynamicAPInt pow(DynamicAPInt x, unsigned y) {
  DynamicAPInt result(1);
  while (y > 0) {
    if (y % 2)
      result *= x;
    x *= x;
    y /= 2;
  }
  return result;
}

std::pair<DynamicAPInt, DynamicAPInt> getBounds(Type type) {
  assert(type.isInteger());
  unsigned width = type.getIntOrFloatBitWidth();
  if (type.isSignedInteger()) {
    auto power = pow(DynamicAPInt(2), width-1);
    return { -power, power-1 };
  }

  auto power = pow(DynamicAPInt(2), width);
  return { DynamicAPInt(0), power-1 };
}

LogicalResult populateConstraint(AffineExpr expr, unsigned numSymbols, Type type, IntegerRelation &constraint) {
  auto maybeSum = extractCoefficients(expr, numSymbols);
  if (!maybeSum)
    return failure();

  IntVector result = *maybeSum;
  // We must guarantee that the result does not overflow.
  // That is, min <= result <= max.
  auto [min, max] = getBounds(type);
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

Value ArithToAffinePass::synthesizeCheck(OpBuilder &builder, Location loc, const IntegerRelation &rel, ValueRange inOperands) {
  // Compute the minimum and maximum for the operands.
  using Bound = std::pair</*min*/DynamicAPInt, /*max*/DynamicAPInt>;
  SmallVector<Bound> bounds;
  SmallVector<Value> operands;

  unsigned numOperands = inOperands.size();
  bounds.reserve(numOperands);
  operands.reserve(numOperands);
  for (auto operand : inOperands) {
    auto cast = llvm::cast<arith::IndexCastOp>(operand.getDefiningOp());
    auto type = cast.getIn().getType();
    bounds.push_back(getBounds(type));
    operands.push_back(cast.getIn());
  }

  const auto synthesize = [&](bool isEq) {
    unsigned numRows = isEq ? rel.getNumEqualities() : rel.getNumInequalities();
    Value condition = arith::ConstantIntOp::create(builder, loc, builder.getI1Type(), 1);
    for (unsigned i = 0; i < numRows; i++) {
      auto row = isEq
        ? rel.getEquality(i)
        : rel.getInequality(i);

      // Compute the maximum and minimum value given the bounds.
      DynamicAPInt min(row.back()), max(row.back());
      for (unsigned j = 0; j < numOperands; j++) {
        min += row[j] * (row[j] < 0 ? bounds[j].second : bounds[j].first);
        max += row[j] * (row[j] < 0 ? bounds[j].first : bounds[j].second);
      }
      int bitWidth = 1 + std::max(log2(min), log2(max));
      Type intType = IntegerType::get(builder.getContext(), bitWidth);
      auto zero = arith::ConstantIntOp::create(builder, loc, intType, 0);
      Value v = zero;
      for (unsigned j = 0; j < numOperands; j++) {
        auto constant = arith::ConstantIntOp::create(builder, loc, intType, (int64_t) row[j]);
        auto ext = arith::ExtSIOp::create(builder, loc, intType, operands[j]);
        auto mul = arith::MulIOp::create(builder, loc, ext, constant);
        v = arith::AddIOp::create(builder, loc, v, mul);
      }
      auto comparison = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sge, v, zero);
      condition = arith::AndIOp::create(builder, loc, condition, comparison);
    }
    return condition;
  };
  Value eqCond = synthesize(true);
  Value ineqCond = synthesize(false);
  return arith::AndIOp::create(builder, loc, eqCond, ineqCond);
}

std::optional<ArithToAffinePass::AffineResult> ArithToAffinePass::constructAffineMap(Value value, const DimIndex &dimIndex) {
  unsigned numSymbols = dimIndex.size();
  PresburgerSpace space = PresburgerSpace::getSetSpace(numSymbols);
  auto universe = IntegerRelation::getUniverse(space);
  MLIRContext *ctx = value.getContext();

  llvm::APInt stepValue;
  if (matchPattern(value, m_ConstantInt(&stepValue))) {
    if (stepValue.getActiveBits() > 64)
      return std::nullopt;

    AffineExpr expr = getAffineConstantExpr(stepValue.getSExtValue(), ctx);
    return AffineResult { expr, universe };
  }

  // We will not trace upwards from block arguments in `valueToAffine()`.
  Operation *def = value.getDefiningOp();
  if (!def) {
    AffineExpr expr = getAffineSymbolExpr(dimIndex.at(value), ctx);
    return AffineResult { expr, universe };
  }

  // So when we reach here, the value must be an operation result.
  // We only consider operations in the arith dialect.
  //
  // We check for binary operations here.
  if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, LLVM::AddOp, LLVM::SubOp, LLVM::MulOp>(def)) {
    auto l = constructAffineMap(def->getOperand(0), dimIndex);
    auto r = constructAffineMap(def->getOperand(1), dimIndex);
    if (!l || !r)
      return std::nullopt;

    AffineExpr expr = llvm::TypeSwitch<Operation*, AffineExpr>(def)
      .Case<arith::AddIOp>([&](arith::AddIOp) { return l->expr + r->expr; })
      .Case<arith::SubIOp>([&](arith::SubIOp) { return l->expr - r->expr; })
      .Case<arith::MulIOp>([&](arith::MulIOp) { return l->expr * r->expr; })
      .Case<LLVM::AddOp>([&](LLVM::AddOp) { return l->expr + r->expr; })
      .Case<LLVM::SubOp>([&](LLVM::SubOp) { return l->expr - r->expr; })
      .Case<LLVM::MulOp>([&](LLVM::MulOp) { return l->expr * r->expr; })
      .DefaultUnreachable("constructAffineMap: should be exhaustive!");

    auto constraint = l->constraint.intersect(r->constraint);
    if (failed(populateConstraint(expr, numSymbols, def->getResult(0).getType(), constraint)))
      return std::nullopt;

    return AffineResult { expr, constraint };
  }
  if (isa<arith::ExtSIOp, arith::ExtUIOp>(def)) {
    // This never imposes any extra constraint.
    return constructAffineMap(def->getOperand(0), dimIndex);
  }

  if (auto gep = dyn_cast<LLVM::GEPOp>(def)) {
    auto indices = gep.getIndices();
    Type elementType = gep.getElemType();
    AffineExpr expr = getAffineConstantExpr(0, ctx);
    IntegerRelation constraint(PresburgerSpace::getSetSpace(numSymbols));
    // We constraint on the sum only.
    DataLayout layout(cast<ModuleOp>(getOperation()));
    for (auto index : indices) {
      if (auto v = dyn_cast<Value>(index)) {
        auto maybe = constructAffineMap(v, dimIndex);
        if (!maybe)
          return std::nullopt;

        const AffineResult &result = *maybe;
        expr = expr + result.expr * layout.getTypeSize(elementType);
        constraint = constraint.intersect(result.constraint);
        
        if (isa<LLVM::LLVMStructType>(elementType))
          return std::nullopt;
        elementType = TypeSwitch<Type, Type>(elementType)
          .Case([](LLVM::LLVMArrayType t) { return t.getElementType(); })
          .Default([](Type t) { return t; });
        // Create the constraint suhc that the entire sum should not overflow.
        if (failed(populateConstraint(expr, numSymbols, v.getType(), constraint)))
          return std::nullopt;
      }
      // TODO: IntegerAttr case
    }
    return AffineResult { expr, constraint };
  }
  return std::nullopt;
}

ArithToAffinePass::AffineMapResult ArithToAffinePass::tidyResult(const AffineResult &result, ValueRange operands) {
  // No need to tidy expressions without operands.
  if (operands.empty())
    return { operands, result.expr, result.constraint };

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

  // Remove columns of constraints.
  unsigned numLeft = used.size();
  IntegerRelation constraint(PresburgerSpace::getSetSpace(numLeft));
  const auto removeColumn = [&](bool isEq) {
    SmallVector<DynamicAPInt> x(numLeft + 1);
    unsigned size = isEq
      ? result.constraint.getNumEqualities()
      : result.constraint.getNumInequalities();
    for (unsigned i = 0; i < size; i++) {
      auto row = isEq
        ? result.constraint.getEquality(i)
        : result.constraint.getInequality(i);
      for (auto [before, after] : indexMap)
        x[after] = row[before];
      x.back() = row.back();
      if (isEq)
        constraint.addEquality(x);
      else
        constraint.addInequality(x);
    }
  };
  removeColumn(true);
  removeColumn(false);

  OpBuilder builder(ctx);
  auto loc = operands[0].getLoc();
  SmallVector<Value> symbols;
  // For every related operand, cast them to index if they aren't already.
  // TODO: deduplicate these casts.
  for (auto [i, v] : llvm::enumerate(operands)) {
    if (!used.contains(i))
      continue;

    if (!isa<IndexType>(v.getType())) {
      builder.setInsertionPointAfterValue(v);
      auto cast = arith::IndexCastOp::create(builder, loc, builder.getIndexType(), v);
      symbols.push_back(cast.getResult());
    } else {
      symbols.push_back(v);
    }
  }

  return AffineMapResult {
    symbols, expr, constraint
  };
}

std::optional<ArithToAffinePass::AffineMapResult> ArithToAffinePass::valueToAffine(Value value) {
  // We trace back the arithmetic computations of the given value till a valid symbol.
  // If eventually we failed to hit anything convertible to symbol, we abort.
  SmallVector<Value> operands;
  collectOperandsUpChain(value, operands);

  // Now compute an affine map and the bounds of the symbols when raising to index.
  // We must guarantee that the index type does not overflow.
  // The arith operations are not necessarily affine, so we do not mutate IR here.
  unsigned numInputs = operands.size();
  DimIndex dimIndex;
  for (auto [i, v] : llvm::enumerate(operands))
    dimIndex[v] = i;
  
  PresburgerSpace space = PresburgerSpace::getSetSpace(numInputs);
  IntegerRelation relation(space);
  
  auto maybeResult = constructAffineMap(value, dimIndex);
  if (!maybeResult)
    return std::nullopt;
  
  return tidyResult(*maybeResult, operands);
}

ArithToAffinePass::LiftResult ArithToAffinePass::lift(Value value) {
  auto liftResult = valueToAffine(value);
  // Give up lifting if we cannot make it affine.
  if (!liftResult)
    return { Value(), Value() };

  // Construct an `affine.apply` with the given symbols and affine map.
  MLIRContext *ctx = value.getContext();
  OpBuilder builder(ctx);
  builder.setInsertionPointAfterValue(value);
  auto loc = value.getLoc();
  auto affineMap = AffineMap::get(0, liftResult->operands.size(), liftResult->expr);
  Value apply = affine::AffineApplyOp::create(builder, loc, builder.getIndexType(), affineMap, liftResult->operands);
  Value condition = synthesizeCheck(builder, loc, liftResult->constraint, liftResult->operands);
  
  return { apply, condition };
}

Value findGEPBase(Value addr) {
  if (auto gep = dyn_cast<LLVM::GEPOp>(addr.getDefiningOp()))
    return findGEPBase(gep.getBase());

  return addr;
}

void ArithToAffinePass::runOnOperation() {
  // Collect loads and stores and attempt to lift them.
  // For now we consider only indexed load/stores. LLVM loads/stores can
  // wait for later.
  SmallVector<Value> tolift;
  getOperation()->walk([&](Operation *op) {
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
        auto branch = scf::IfOp::create(builder, loc, result.condition, [&](OpBuilder &builder, Location loc) {
          auto gepBase = findGEPBase(addr);
          auto cast = arith::IndexCastOp::create(builder, loc, builder.getI64Type(), result.apply);
          auto gep = LLVM::GEPOp::create(builder, loc, addr.getType(), builder.getI8Type(), gepBase, {cast});
          scf::YieldOp::create(builder, loc, gep.getResult());
        }, [&](OpBuilder &builder, Location loc) {
          scf::YieldOp::create(builder ,loc, addr);
        });
        load.setOperand(branch.getResult(0));
      }
    }
  }
}

} // namespace
