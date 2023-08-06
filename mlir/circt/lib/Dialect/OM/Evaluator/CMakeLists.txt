add_circt_library(CIRCTOMEvaluator
  Evaluator.cpp

  DEPENDS
  MLIROMIncGen
  MLIROMAttrIncGen
  MLIROMOpInterfacesIncGen

  LINK_LIBS
  CIRCTOM
  MLIRIR
)
