add_circt_translation_library(CIRCTImportFIRFile
  FIRAnnotations.cpp
  FIRLexer.cpp
  FIRParser.cpp
  FIRParserAsserts.cpp

  ADDITIONAL_HEADER_DIRS

  LINK_LIBS PUBLIC
  CIRCTFIRRTL
  CIRCTFIRRTLTransforms
  MLIRTranslateLib
  )
