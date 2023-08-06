add_circt_translation_library(CIRCTExportChiselInterface
  ExportChiselInterface.cpp

  ADDITIONAL_HEADER_DIRS

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTFIRRTL
  MLIRPass
  MLIRSideEffectInterfaces
)
