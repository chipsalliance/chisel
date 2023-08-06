add_circt_dialect(Interop interop)
add_circt_dialect_doc(Interop interop)

set(LLVM_TARGET_DEFINITIONS Interop.td)

mlir_tablegen(InteropEnums.h.inc -gen-enum-decls)
mlir_tablegen(InteropEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(CIRCTInteropEnumsIncGen)
add_dependencies(circt-headers CIRCTInteropEnumsIncGen)
