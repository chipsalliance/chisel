add_circt_dialect_library(CIRCTSystemCTransforms
  SystemCLowerInstanceInterop.cpp

  DEPENDS
  CIRCTSystemCTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTSystemC
  CIRCTInteropDialect
  CIRCTSupport
  MLIRIR
  MLIRFuncDialect
  MLIRPass
  MLIRTransformUtils
)
