add_circt_conversion_library(CIRCTVerifToSV
  VerifToSV.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTSV
  CIRCTVerif
  MLIRTransforms
)
