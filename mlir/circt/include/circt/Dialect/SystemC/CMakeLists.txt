add_circt_dialect(SystemC systemc)
add_circt_dialect_doc(SystemC systemc)

set(LLVM_TARGET_DEFINITIONS SystemC.td)

mlir_tablegen(SystemCEnums.h.inc -gen-enum-decls)
mlir_tablegen(SystemCEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(CIRCTSystemCEnumsIncGen)
add_dependencies(circt-headers CIRCTSystemCEnumsIncGen)

mlir_tablegen(SystemCAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(SystemCAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(CIRCTSystemCAttributesIncGen)
add_dependencies(circt-headers CIRCTSystemCAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS SystemCOpInterfaces.td)
mlir_tablegen(SystemCOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(SystemCOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(CIRCTSystemCOpInterfacesIncGen)
add_dependencies(circt-headers CIRCTSystemCOpInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(CIRCTSystemCTransformsIncGen)
add_circt_doc(Passes SystemCPasses -gen-pass-doc)
