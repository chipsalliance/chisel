
add_circt_translation_library(CIRCTExportVerilog
  ApplyLoweringOptions.cpp
  ExportVerilog.cpp
  LegalizeAnonEnums.cpp
  LegalizeNames.cpp
  PrepareForEmission.cpp
  PruneZeroValuedLogic.cpp

  ADDITIONAL_HEADER_DIRS

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTHW
  CIRCTLTL
  CIRCTOM
  CIRCTSupport
  CIRCTSV
  CIRCTSupport
  CIRCTVerif
  MLIRIR
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRTransforms
)
