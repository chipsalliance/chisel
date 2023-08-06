set(LLVM_LINK_COMPONENTS Support)

add_llvm_tool(arcilator arcilator.cpp)
target_link_libraries(arcilator
  PRIVATE
  CIRCTArc
  CIRCTArcToLLVM
  CIRCTArcTransforms
  CIRCTCombToArith
  CIRCTConvertToArcs
  CIRCTSeqTransforms
  CIRCTSupport
  CIRCTTransforms
  MLIRParser
  MLIRLLVMIRTransforms
  MLIRTargetLLVMIRExport
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRFuncInlinerExtension
)

llvm_update_compile_flags(arcilator)
mlir_check_all_link_libraries(arcilator)

configure_file(arcilator-header-cpp.py
  ${CIRCT_TOOLS_DIR}/arcilator-header-cpp.py)
add_custom_target(arcilator-header-cpp SOURCES
  ${CIRCT_TOOLS_DIR}/arcilator-header-cpp.py)

configure_file(arcilator-runtime.h
  ${CIRCT_TOOLS_DIR}/arcilator-runtime.h)
add_custom_target(arcilator-runtime-header SOURCES
  ${CIRCT_TOOLS_DIR}/arcilator-runtime.h)
