add_circt_library(CIRCTPipelineToHW
  PipelineToHW.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  CIRCTSeq
  CIRCTHW
  CIRCTComb
  CIRCTPipelineOps
  )
