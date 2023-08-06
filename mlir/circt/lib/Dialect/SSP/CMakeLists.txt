##===- CMakeLists.txt - SSP dialect code ----------------------*- cmake -*-===//
##
## Implementation files for the SSP dialect.
##
##===----------------------------------------------------------------------===//

set(SSP_Srcs
  SSPAttributes.cpp
  SSPDialect.cpp
  SSPOps.cpp
  )

set(SSP_LinkLibs
  CIRCTScheduling
  CIRCTSupport
  MLIRIR
  )

set(SSP_Deps
  ${SSP_LinkLibs}
  CIRCTSSPAttributesIncGen
  )

add_circt_dialect_library(CIRCTSSP
  ${SSP_Srcs}

  DEPENDS
  ${SSP_Deps}

  LINK_COMPONENTS
  Core
  Support

  LINK_LIBS PUBLIC
  ${SSP_LinkLibs}
  )

add_subdirectory(Transforms)
