add_circt_dialect(LTL ltl)
add_circt_doc(LTLOps Dialects/LTLOps -gen-op-doc)
add_circt_doc(LTLTypes Dialects/LTLTypes -gen-typedef-doc -dialect ltl)

set(LLVM_TARGET_DEFINITIONS LTL.td)
mlir_tablegen(LTLEnums.h.inc -gen-enum-decls)
mlir_tablegen(LTLEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(CIRCTLTLEnumsIncGen)
add_dependencies(circt-headers CIRCTLTLEnumsIncGen)

mlir_tablegen(LTLFolds.cpp.inc -gen-rewriters)
add_public_tablegen_target(CIRCTLTLFoldsIncGen)
add_dependencies(circt-headers CIRCTLTLFoldsIncGen)
