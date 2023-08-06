add_circt_conversion_library(CIRCTHWToSystemC
  HWToSystemC.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTSystemC
  CIRCTHW
  CIRCTComb
  MLIREmitCDialect
  MLIRTransforms
)
