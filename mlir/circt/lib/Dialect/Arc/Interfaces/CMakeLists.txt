add_circt_dialect_library(CIRCTArcExternalInterfaces
  RuntimeCostEstimateInterfaceImpl.cpp

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTComb
  MLIRIR
  MLIRSCFDialect
)
