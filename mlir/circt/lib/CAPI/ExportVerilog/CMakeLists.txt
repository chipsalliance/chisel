add_mlir_public_c_api_library(CIRCTCAPIExportVerilog
  ExportVerilog.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  CIRCTExportVerilog
  )
