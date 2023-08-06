add_circt_conversion_library(CIRCTLLHDToLLVM
  LLHDToLLVM.cpp

  DEPENDS
  CIRCTConversionPassIncGen
  MLIRArithDialect

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTLLHD
  CIRCTComb
  CIRCTCombToArith
  CIRCTCombToLLVM
  CIRCTHWToLLVM
  CIRCTHW
  CIRCTSupport
  MLIRArithToLLVM
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRLLVMCommonConversion
  MLIRVectorDialect
  MLIRTransforms
  MLIRReconcileUnrealizedCasts
  )
