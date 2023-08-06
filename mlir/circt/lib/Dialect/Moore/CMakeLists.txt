add_circt_dialect_library(CIRCTMoore
  MooreDialect.cpp
  MooreTypes.cpp
  MIROps.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/Moore

  DEPENDS
  MLIRMooreIncGen
  CIRCTMooreAttributesIncGen
  CIRCTMooreEnumsIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
)

add_dependencies(circt-headers MLIRMooreIncGen)
