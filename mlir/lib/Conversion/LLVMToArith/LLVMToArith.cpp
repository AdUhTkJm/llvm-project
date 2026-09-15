//===- LLVMToArith.cpp - Convert LLVM to Arith dialect --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMToArith/LLVMToArith.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTLLVMTOARITHPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

/// Convert the LLVM dialect integer overflow flags to their arith dialect
/// counterparts.
static arith::IntegerOverflowFlags
convertOverflowFlags(LLVM::IntegerOverflowFlags flags) {
  arith::IntegerOverflowFlags result = arith::IntegerOverflowFlags::none;
  if (bitEnumContainsAny(flags, LLVM::IntegerOverflowFlags::nsw))
    result = result | arith::IntegerOverflowFlags::nsw;
  if (bitEnumContainsAny(flags, LLVM::IntegerOverflowFlags::nuw))
    result = result | arith::IntegerOverflowFlags::nuw;
  return result;
}

/// Convert the LLVM dialect integer comparison predicate to the arith
/// dialect equivalent.
static arith::CmpIPredicate convertPredicate(LLVM::ICmpPredicate pred) {
  switch (pred) {
  case LLVM::ICmpPredicate::eq:
    return arith::CmpIPredicate::eq;
  case LLVM::ICmpPredicate::ne:
    return arith::CmpIPredicate::ne;
  case LLVM::ICmpPredicate::slt:
    return arith::CmpIPredicate::slt;
  case LLVM::ICmpPredicate::sle:
    return arith::CmpIPredicate::sle;
  case LLVM::ICmpPredicate::sgt:
    return arith::CmpIPredicate::sgt;
  case LLVM::ICmpPredicate::sge:
    return arith::CmpIPredicate::sge;
  case LLVM::ICmpPredicate::ult:
    return arith::CmpIPredicate::ult;
  case LLVM::ICmpPredicate::ule:
    return arith::CmpIPredicate::ule;
  case LLVM::ICmpPredicate::ugt:
    return arith::CmpIPredicate::ugt;
  case LLVM::ICmpPredicate::uge:
    return arith::CmpIPredicate::uge;
  }
  llvm_unreachable("unknown LLVM icmp predicate");
}

/// Convert an LLVM dialect binary integer operation with overflow flags
/// (add, sub, mul) to the corresponding arith dialect operation, preserving
/// the overflow flags.
template <typename LLVMOp, typename ArithOp>
struct BinOpWithOverflowFlagsLowering : public OpRewritePattern<LLVMOp> {
  using OpRewritePattern<LLVMOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVMOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ArithOp>(
        op, op.getType(), op.getLhs(), op.getRhs(),
        convertOverflowFlags(op.getOverflowFlags()));
    return success();
  }
};

/// Convert an LLVM dialect integer division operation (sdiv, udiv) to the
/// corresponding arith dialect operation, preserving the exactness flag.
template <typename LLVMOp, typename ArithOp>
struct DivOpLowering : public OpRewritePattern<LLVMOp> {
  using OpRewritePattern<LLVMOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVMOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ArithOp>(op, op.getType(), op.getLhs(),
                                         op.getRhs(), op.getIsExact());
    return success();
  }
};

/// Return true if the operands of the given `llvm.icmp` are integer-like and
/// the operation can therefore be converted to `arith.cmpi`. Comparisons of
/// pointers (or vectors thereof) are not representable in the arith dialect
/// and must remain in the LLVM dialect.
static bool isConvertibleToCmpI(LLVM::ICmpOp op) {
  Type elemType = getElementTypeOrSelf(op.getLhs().getType());
  return isa<IntegerType, IndexType>(elemType);
}

/// Convert `llvm.icmp` to `arith.cmpi`.
struct ICmpOpLowering : public OpRewritePattern<LLVM::ICmpOp> {
  using OpRewritePattern<LLVM::ICmpOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::ICmpOp op,
                                PatternRewriter &rewriter) const override {
    if (!isConvertibleToCmpI(op))
      return failure();
    rewriter.replaceOpWithNewOp<arith::CmpIOp>(
        op, convertPredicate(op.getPredicate()), op.getLhs(), op.getRhs());
    return success();
  }
};

struct ConvertLLVMToArithPass
    : public impl::ConvertLLVMToArithPassBase<ConvertLLVMToArithPass> {
  using Base::Base;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateLLVMToArithConversionPatterns(patterns);

    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect, LLVM::LLVMDialect>();
    target.addIllegalOp<LLVM::AddOp, LLVM::SubOp, LLVM::MulOp, LLVM::SDivOp,
                        LLVM::UDivOp>();
    // `llvm.icmp` also supports pointer comparisons, which have no `arith`
    // counterpart; keep those in the LLVM dialect.
    target.addDynamicallyLegalOp<LLVM::ICmpOp>(
        [](LLVM::ICmpOp op) { return !isConvertibleToCmpI(op); });
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void mlir::populateLLVMToArithConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<BinOpWithOverflowFlagsLowering<LLVM::AddOp, arith::AddIOp>,
               BinOpWithOverflowFlagsLowering<LLVM::SubOp, arith::SubIOp>,
               BinOpWithOverflowFlagsLowering<LLVM::MulOp, arith::MulIOp>,
               DivOpLowering<LLVM::SDivOp, arith::DivSIOp>,
               DivOpLowering<LLVM::UDivOp, arith::DivUIOp>, ICmpOpLowering>(
      patterns.getContext());
}
