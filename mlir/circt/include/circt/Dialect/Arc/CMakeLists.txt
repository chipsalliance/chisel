add_circt_dialect(Arc arc)
add_circt_dialect_doc(Arc arc)

set(LLVM_TARGET_DEFINITIONS ArcPasses.td)
mlir_tablegen(ArcPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(CIRCTArcTransformsIncGen)
add_circt_doc(ArcPasses ArcPasses -gen-pass-doc)

set(LLVM_TARGET_DEFINITIONS Arc.td)
mlir_tablegen(ArcEnums.h.inc -gen-enum-decls)
mlir_tablegen(ArcEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(CIRCTArcEnumsIncGen)
add_dependencies(circt-headers CIRCTArcEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS ArcInterfaces.td)
mlir_tablegen(ArcInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(ArcInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(CIRCTArcInterfacesIncGen)
add_dependencies(circt-headers CIRCTArcInterfacesIncGen)
