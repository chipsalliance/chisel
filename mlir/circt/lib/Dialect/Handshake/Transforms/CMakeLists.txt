add_circt_dialect_library(CIRCTHandshakeTransforms
  Analysis.cpp
  PassHelpers.cpp
  Materialization.cpp
  Buffers.cpp
  LockFunctions.cpp
  LowerExtmemToHW.cpp
  LegalizeMemrefs.cpp

  DEPENDS
  CIRCTHandshakeTransformsIncGen
  MLIRArithDialect

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTESI
  CIRCTHandshake
  CIRCTSupport
  CIRCTTransforms
  MLIRIR
  MLIRSCFDialect
  MLIRPass
  MLIRTransformUtils
  MLIRMemRefDialect
  MLIRAffineDialect
  )
