##===- CMakeLists.txt - Calyx dialect code --------------------*- cmake -*-===//
##
## Implementation files for the Calyx dialect.
##
##===----------------------------------------------------------------------===//

add_circt_dialect_library(CIRCTCalyx
  CalyxDialect.cpp
  CalyxOps.cpp

  DEPENDS
  MLIRCalyxEnumsIncGen
  MLIRCalyxInterfacesIncGen

  LINK_COMPONENTS
  Core
  Support

  LINK_LIBS PUBLIC
  CIRCTSupport
  CIRCTComb
  CIRCTSV
  CIRCTHW
  MLIRArithDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRTransforms
  )

add_subdirectory(Export)
add_subdirectory(Transforms)
