add_circt_dialect_library(CIRCTMSFTTransforms
  MSFTExportTcl.cpp
  MSFTLowerInstances.cpp
  MSFTPassCommon.cpp
  MSFTWireCleanup.cpp
  MSFTDiscoverAppIDs.cpp
  MSFTLowerConstructs.cpp
  MSFTPartition.cpp
  MSFTPassCommon.h
  MSFTToHW.cpp
  PassCommon.cpp

  DEPENDS
  CIRCTMSFTTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTMSFT
  CIRCTHW
  CIRCTSV
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
