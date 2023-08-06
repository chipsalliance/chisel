add_circt_dialect(SV sv)
add_circt_dialect_doc(SV sv)

set(LLVM_TARGET_DEFINITIONS SV.td)

mlir_tablegen(SVEnums.h.inc -gen-enum-decls)
mlir_tablegen(SVEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRSVEnumsIncGen)
add_dependencies(circt-headers MLIRSVEnumsIncGen)

mlir_tablegen(SVAttributes.h.inc -gen-attrdef-decls --attrdefs-dialect=sv)
mlir_tablegen(SVAttributes.cpp.inc -gen-attrdef-defs --attrdefs-dialect=sv)
add_public_tablegen_target(MLIRSVAttributesIncGen)
add_dependencies(circt-headers MLIRSVAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS SVPasses.td)
mlir_tablegen(SVPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(CIRCTSVTransformsIncGen)
add_circt_doc(SVPasses SVPasses -gen-pass-doc)
