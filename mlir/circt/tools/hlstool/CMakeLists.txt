set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_tool(hlstool
 hlstool.cpp
)
llvm_update_compile_flags(hlstool)
target_link_libraries(hlstool
  PRIVATE

  CIRCTESI
  CIRCTExportChiselInterface
  CIRCTExportVerilog
  CIRCTHandshake
  CIRCTHandshakeToHW
  CIRCTHandshakeTransforms
  CIRCTHW
  CIRCTHWTransforms
  CIRCTSeq
  CIRCTSeqTransforms
  CIRCTStandardToHandshake
  CIRCTSV
  CIRCTSVTransforms
  CIRCTTransforms
  
  MLIRIR
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIROptLib
  MLIRParser
  MLIRControlFlowDialect
  MLIRSupport
  MLIRTransforms
  MLIRSCFToControlFlow
  )
