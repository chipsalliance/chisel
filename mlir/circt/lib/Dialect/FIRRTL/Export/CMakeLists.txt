add_circt_translation_library(CIRCTExportFIRRTL
  FIREmitter.cpp

  LINK_LIBS PUBLIC
  CIRCTFIRRTL
  MLIRTranslateLib
)
