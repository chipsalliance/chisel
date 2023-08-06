add_circt_dialect_library(CIRCTSV
  SVAttributes.cpp
  SVDialect.cpp
  SVOps.cpp
  SVTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/SV

  DEPENDS
  MLIRSVIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTHW
  CIRCTSupport
  MLIRIR
)

add_dependencies(circt-headers MLIRSVIncGen)

add_subdirectory(Transforms)
