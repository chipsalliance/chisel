add_circt_conversion_library(CIRCTHWToSV
  HWToSV.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTSV
  MLIREmitCDialect
  MLIRTransforms
)
