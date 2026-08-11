#include "mlir/Conversion/MemRefToArith/MemRefToArith.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTMEMREFTOARITH
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

// The name is bad. It mainly reconciles the unrealized casts caused
// by memref -> llvm conversion and interaction with arith.
struct MemRefToArithPass : impl::ConvertMemRefToArithBase<MemRefToArithPass> {
  using Base::Base;

  void runOnOperation() override;
};

struct RealizeIndexToI64Cast : public mlir::OpRewritePattern<mlir::UnrealizedConversionCastOp> {
  using OpRewritePattern<mlir::UnrealizedConversionCastOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::UnrealizedConversionCastOp op,
      mlir::PatternRewriter &rewriter) const override {
    
    if (op.getInputs().size() != 1 || op.getOutputs().size() != 1)
      return mlir::failure();

    Value in = op.getInputs()[0];
    if (!in.getDefiningOp())
      return mlir::failure();
    auto indexCast = dyn_cast<arith::IndexCastOp>(in.getDefiningOp());
    if (!indexCast)
      return mlir::failure();
    
    mlir::Type inType = in.getType();
    mlir::Type outType = op.getOutputs()[0].getType();

    if (inType.isIndex() && outType.isInteger()) {
      rewriter.replaceOpWithNewOp<mlir::arith::ExtUIOp>(op, outType, indexCast.getIn());
      return mlir::success();
    }

    return mlir::failure();
  }
};

void MemRefToArithPass::runOnOperation() {
  MLIRContext &ctx = getContext();
  RewritePatternSet patterns(&ctx);
  populateMemRefToArithConversionPatterns(patterns);
  
  ConversionTarget target(getContext());

  target.addLegalDialect<arith::ArithDialect>();
  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

} // namespace

void mlir::populateMemRefToArithConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<RealizeIndexToI64Cast>(patterns.getContext());
}
