add_circt_dialect(Moore moore)
add_circt_dialect_doc(Moore moore)

set(LLVM_TARGET_DEFINITIONS Moore.td)

mlir_tablegen(MooreEnums.h.inc -gen-enum-decls)
mlir_tablegen(MooreEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(CIRCTMooreEnumsIncGen)
add_dependencies(circt-headers CIRCTMooreEnumsIncGen)

mlir_tablegen(MooreAttributes.h.inc -gen-attrdef-decls
  -attrdefs-dialect MooreDialect)
mlir_tablegen(MooreAttributes.cpp.inc -gen-attrdef-defs
  -attrdefs-dialect MooreDialect)
add_public_tablegen_target(CIRCTMooreAttributesIncGen)
add_dependencies(circt-headers CIRCTMooreAttributesIncGen)
