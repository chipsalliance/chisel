add_circt_dialect_library(CIRCTLTL
  LTLDialect.cpp
  LTLFolds.cpp
  LTLOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/LTL

  DEPENDS
  CIRCTLTLEnumsIncGen
  MLIRLTLIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTHW
  MLIRIR
  MLIRInferTypeOpInterface
)
