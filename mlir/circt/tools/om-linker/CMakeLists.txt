set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_tool(om-linker
  om-linker.cpp
)
llvm_update_compile_flags(om-linker)
target_link_libraries(om-linker PRIVATE
  CIRCTOM
  CIRCTOMTransforms
  CIRCTSupport

  MLIRBytecodeReader
  MLIRBytecodeWriter
  MLIRParser
  MLIRSupport
  MLIRIR
  MLIROptLib
)
