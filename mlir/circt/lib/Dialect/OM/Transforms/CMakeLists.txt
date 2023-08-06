add_circt_dialect_library(CIRCTOMTransforms
  LinkModules.cpp

  DEPENDS
  CIRCTOMTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTOM
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
