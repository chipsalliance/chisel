add_circt_dialect_library(CIRCTVerif
  VerifDialect.cpp
  VerifOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/Verif

  DEPENDS
  MLIRVerifIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  CIRCTLTL
  MLIRIR
)
