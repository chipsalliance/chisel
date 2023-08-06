add_circt_conversion_library(CIRCTHWToLLHD
  HWToLLHD.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTLLHD
  CIRCTHW
  MLIRTransforms
)
