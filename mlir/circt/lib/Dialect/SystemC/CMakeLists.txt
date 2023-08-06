add_circt_dialect_library(CIRCTSystemC
  SystemCAttributes.cpp
  SystemCDialect.cpp
  SystemCOpInterfaces.cpp
  SystemCOps.cpp
  SystemCTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/SystemC

  DEPENDS
  MLIRSystemCIncGen
  CIRCTSystemCAttributesIncGen
  CIRCTSystemCEnumsIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIREmitCDialect
  CIRCTHW
  CIRCTSupport
)

add_dependencies(circt-headers MLIRSystemCIncGen)

add_subdirectory(Transforms)
