#include "mlir/Conversion/LLVMToControlFlow/LLVMToControlFlow.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTLLVMTOCONTROLFLOWPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::LLVM;

namespace {

struct LLVMToControlFlowPass
    : public impl::ConvertLLVMToControlFlowPassBase<LLVMToControlFlowPass> {
  using Base::Base;
  void runOnOperation() override;
};

struct BrLifting : public OpRewritePattern<BrOp> {
  using OpRewritePattern<BrOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BrOp br,
                                PatternRewriter &rewriter) const override;
};

LogicalResult BrLifting::matchAndRewrite(BrOp br, PatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<cf::BranchOp>(br, br.getDest(), br.getDestOperands());
  return success();
}

struct CondBrLifting : public OpRewritePattern<CondBrOp> {
  using OpRewritePattern<CondBrOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CondBrOp br,
                                PatternRewriter &rewriter) const override;
};

LogicalResult CondBrLifting::matchAndRewrite(CondBrOp condbr, PatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<cf::CondBranchOp>(condbr, condbr.getCondition(), condbr.getTrueDest(), condbr.getTrueDestOperands(), condbr.getFalseDest(), condbr.getFalseDestOperands(), condbr.getWeights());
  return success();
}

void LLVMToControlFlowPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateLLVMToControlFlowConversionPatterns(patterns);
  
  ConversionTarget target(getContext());

  target.addLegalDialect<cf::ControlFlowDialect>();
  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

} // namespace

void mlir::populateLLVMToControlFlowConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<BrLifting, CondBrLifting>(patterns.getContext());
}
