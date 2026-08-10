#include "mlir/Conversion/ArithToAffine/ArithToAffine.h"

#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemoryAccessOpInterfaces.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "llvm/ADT/DynamicAPInt.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTARITHTOAFFINE
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::presburger;

namespace {

struct ArithToAffinePass : impl::ConvertArithToAffineBase<ArithToAffinePass> {
  using Base::Base;

  struct LiftResult {
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
    // Indices of unused operands. For example, this could include the const
    // values that are folded already.
    DenseSet<int> unused;
  };

  // Maps each affine value to its dimension index in a presburger relation.
  using DimIndex = llvm::DenseMap<Value, int>;

  std::optional<LiftResult> valueToAffine(Value value);
  std::optional<AffineResult> constructAffineMap(Value value, const DimIndex &dimIndex);
  Value lift(Value value);
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

  if (isa<arith::ArithDialect>(def->getDialect())) {
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

template<class T>
DenseSet<T> intersect(const DenseSet<T> &a, const DenseSet<T> &b) {
  DenseSet<T> result;
  for (auto x : a) {
    if (b.contains(x))
      result.insert(x);
  }
  return result;
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
    DenseSet<int> unused { dimIndex.at(value) };
    return AffineResult { expr, universe, unused };
  }

  // We will not trace upwards from block arguments in `valueToAffine()`.
  Operation *def = value.getDefiningOp();
  if (!def) {
    AffineExpr expr = getAffineSymbolExpr(dimIndex.at(value), ctx);
    return AffineResult { expr, universe, DenseSet<int>() };
  }

  // So when we reach here, the value must be an operation result.
  // We only consider operations in the arith dialect.
  //
  // We check for binary operations here.
  if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp>(def)) {
    auto l = constructAffineMap(def->getOperand(0), dimIndex);
    auto r = constructAffineMap(def->getOperand(1), dimIndex);
    if (!l || !r)
      return std::nullopt;

    AffineExpr expr = llvm::TypeSwitch<Operation*, AffineExpr>(def)
      .Case<arith::AddIOp>([&](arith::AddIOp) { return l->expr + r->expr; })
      .Case<arith::SubIOp>([&](arith::SubIOp) { return l->expr - r->expr; })
      .Case<arith::MulIOp>([&](arith::MulIOp) { return l->expr * r->expr; })
      .DefaultUnreachable("constructAffineMap: only accepts addi and muli!");

    auto constraint = l->constraint.intersect(r->constraint);
    auto maybeSum = extractCoefficients(expr, numSymbols);
    if (!maybeSum)
      return std::nullopt;

    IntVector result = *maybeSum;
    // We must guarantee that the result does not overflow.
    // That is, min <= result <= max.
    auto [min, max] = getBounds(def->getResult(0).getType());
    llvm::interleaveComma(result, llvm::errs()); llvm::errs() << "\n";
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
    return AffineResult { expr, constraint, intersect(l->unused, r->unused) };
  }
  return std::nullopt;
}

std::optional<ArithToAffinePass::LiftResult> ArithToAffinePass::valueToAffine(Value value) {
  OpBuilder builder(value.getContext());
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
  
  auto result = constructAffineMap(value, dimIndex);
  if (!result)
    return std::nullopt;
  
  auto loc = value.getLoc();
  SmallVector<Value> symbols;
  // For every related operand, cast them to index if they aren't already.
  // TODO: deduplicate these casts.
  for (auto [i, v] : llvm::enumerate(operands)) {
    if (result->unused.contains(i))
      continue;

    if (!isa<IndexType>(v.getType())) {
      builder.setInsertionPointAfterValue(v);
      auto cast = arith::IndexCastOp::create(builder, loc, builder.getIndexType(), v);
      symbols.push_back(cast.getResult());
    } else {
      symbols.push_back(v);
    }
  }

  return LiftResult {
    symbols, result->expr, result->constraint
  };
}

Value ArithToAffinePass::lift(Value value) {
  auto liftResult = valueToAffine(value);
  // Give up lifting if we cannot make it affine.
  if (!liftResult)
    return Value();

  // Construct an `affine.apply` with the given symbols and affine map.
  MLIRContext *ctx = value.getContext();
  OpBuilder builder(ctx);
  builder.setInsertionPointAfterValue(value);
  auto loc = value.getLoc();
  auto affineMap = AffineMap::get(0, liftResult->operands.size(), liftResult->expr);
  auto apply = affine::AffineApplyOp::create(builder, loc, builder.getIndexType(), affineMap, liftResult->operands);
  llvm::errs() << "constraint:\n";
  liftResult->constraint.dump();
  llvm::errs() << "\n";

  return apply;
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
  });

  for (auto value : tolift) {
    if (auto cast = dyn_cast<arith::IndexCastOp>(value.getDefiningOp())) {
      if (auto apply = lift(cast.getOperand()))
        value.replaceAllUsesWith(apply);
    }
  }
}

} // namespace
