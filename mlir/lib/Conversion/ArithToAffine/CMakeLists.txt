add_mlir_conversion_library(MLIRArithToAffine
  ArithToAffine.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/ArithToAffine

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRArithDialect
  MLIRArithUtils
  MLIRPass
  MLIRSCFDialect
  MLIRTransforms
  MLIRValueBoundsOpInterface
  )
