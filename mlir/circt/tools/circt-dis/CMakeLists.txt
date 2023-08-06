set(LLVM_LINK_COMPONENTS
  Support
)

get_property(dialect_libs GLOBAL PROPERTY CIRCT_DIALECT_LIBS)
get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set(LIBS
  ${dialect_libs}
  ${mlir_dialect_libs}

  MLIRIR
  MLIRBytecodeReader
  MLIRParser
  MLIRSupport
)

add_llvm_tool(circt-dis circt-dis.cpp DEPENDS ${LIBS})
target_link_libraries(circt-dis PRIVATE ${LIBS})

llvm_update_compile_flags(circt-dis)
mlir_check_all_link_libraries(circt-dis)
