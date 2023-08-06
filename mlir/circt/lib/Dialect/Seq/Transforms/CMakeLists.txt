add_circt_dialect_library(CIRCTSeqTransforms
  ExternalizeClockGate.cpp
  LowerFirMem.cpp
  LowerSeqHLMem.cpp
  LowerSeqToSV.cpp
  LowerSeqFIFO.cpp

  DEPENDS
  CIRCTSeqTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTHW
  CIRCTSeq
  CIRCTSupport
  CIRCTSV
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
