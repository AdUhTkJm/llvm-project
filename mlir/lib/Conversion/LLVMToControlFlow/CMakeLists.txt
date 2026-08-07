add_mlir_conversion_library(MLIRLLVMToCF
  LLVMToControlFlow.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/LLVMToControlFlow

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRLLVMDialect
  MLIRControlFlowDialect
  MLIRPass
  MLIRTransformUtils
)
