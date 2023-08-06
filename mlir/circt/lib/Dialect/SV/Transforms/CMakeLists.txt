add_circt_dialect_library(CIRCTSVTransforms
  GeneratorCallout.cpp
  HWCleanup.cpp
  HWStubExternalModules.cpp
  HWLegalizeModules.cpp
  HWMemSimImpl.cpp
  PrettifyVerilog.cpp
  SVExtractTestCode.cpp
  HWExportModuleHierarchy.cpp
  SVTraceIVerilog.cpp
  HWEliminateInOutPorts.cpp

  DEPENDS
  CIRCTSVTransformsIncGen

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
