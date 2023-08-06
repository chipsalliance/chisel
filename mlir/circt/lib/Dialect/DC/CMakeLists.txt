add_circt_dialect_library(CIRCTDC
  DCDialect.cpp
  DCOps.cpp
  DCTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/DC

  DEPENDS
  MLIRArithDialect
  MLIRDCIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  MLIRIR
  CIRCTESI
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
)

add_dependencies(circt-headers
  MLIRDCIncGen
)

add_subdirectory(Transforms)
