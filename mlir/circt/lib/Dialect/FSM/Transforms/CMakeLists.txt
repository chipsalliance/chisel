add_circt_dialect_library(CIRCTFSMTransforms
  PrintFSMGraph.cpp

  DEPENDS
  CIRCTFSMTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTFSM
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
