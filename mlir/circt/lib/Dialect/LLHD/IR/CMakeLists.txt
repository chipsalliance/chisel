add_circt_dialect_library(CIRCTLLHD
  LLHDDialect.cpp
  LLHDOps.cpp
  LLHDTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Dialect/LLHD

  DEPENDS
  CIRCTLLHDEnumsIncGen
  CIRCTLLHDAttributesIncGen
  MLIRLLHDIncGen
  
  LINK_LIBS PUBLIC
  CIRCTHW
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRControlFlowInterfaces
  MLIRInferTypeOpInterface
  MLIRCallInterfaces
  MLIRFuncDialect
)
