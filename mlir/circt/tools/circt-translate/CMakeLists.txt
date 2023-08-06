get_property(dialect_libs GLOBAL PROPERTY CIRCT_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY CIRCT_TRANSLATION_LIBS)

set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_tool(circt-translate
  circt-translate.cpp
)

llvm_update_compile_flags(circt-translate)

target_link_libraries(circt-translate
  PRIVATE
  ${dialect_libs}
  ${translation_libs}
  MLIRIR
  MLIRSupport
  MLIRTranslateLib
  )
