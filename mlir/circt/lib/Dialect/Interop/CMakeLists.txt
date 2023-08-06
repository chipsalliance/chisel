add_circt_dialect_library(CIRCTInteropDialect
  InteropDialect.cpp
  InteropOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/Interop

  DEPENDS
  MLIRInteropIncGen
  CIRCTInteropEnumsIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  MLIRIR
)

add_dependencies(circt-headers MLIRInteropIncGen)
