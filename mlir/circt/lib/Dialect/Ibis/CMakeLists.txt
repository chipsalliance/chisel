add_circt_dialect_library(CIRCTIbis
  IbisDialect.cpp
  IbisOps.cpp
  IbisTypes.cpp

  DEPENDS
  MLIRIbisIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  CIRCTHW
  )

add_subdirectory(Transforms)
