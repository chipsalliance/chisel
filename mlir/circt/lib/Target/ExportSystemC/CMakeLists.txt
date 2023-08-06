add_circt_translation_library(CIRCTExportSystemC
  EmissionPrinter.cpp
  ExportSystemC.cpp
  Patterns/BuiltinEmissionPatterns.cpp
  Patterns/EmitCEmissionPatterns.cpp
  Patterns/HWEmissionPatterns.cpp
  Patterns/SystemCEmissionPatterns.cpp

  ADDITIONAL_HEADER_DIRS

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTHW
  CIRCTSystemC
  CIRCTSupport
  MLIREmitCDialect
  MLIRIR
  MLIRTranslateLib
)
