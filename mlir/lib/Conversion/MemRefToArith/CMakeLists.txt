add_mlir_conversion_library(MLIRMemRefToArith
  MemRefToArith.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/MemRefToArith

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRArithUtils
  MLIRPass
  MLIRTransforms
  )
