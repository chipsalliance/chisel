add_circt_dialect(HW hw)

set(LLVM_TARGET_DEFINITIONS HW.td)

mlir_tablegen(HWAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(HWAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRHWAttrIncGen)
add_dependencies(circt-headers MLIRHWAttrIncGen)

mlir_tablegen(HWEnums.h.inc -gen-enum-decls)
mlir_tablegen(HWEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRHWEnumsIncGen)
add_dependencies(circt-headers MLIRHWEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(CIRCTHWTransformsIncGen)

set(LLVM_TARGET_DEFINITIONS HWOpInterfaces.td)
mlir_tablegen(HWOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(HWOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(CIRCTHWOpInterfacesIncGen)
add_dependencies(circt-headers CIRCTHWOpInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS HWTypeInterfaces.td)
mlir_tablegen(HWTypeInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(HWTypeInterfaces.cpp.inc -gen-type-interface-defs)
add_public_tablegen_target(CIRCTHWTypeInterfacesIncGen)
add_dependencies(circt-headers CIRCTHWTypeInterfacesIncGen)

# Generate Dialect documentation.
add_circt_doc(HWAggregates Dialects/HWAggregateOps -gen-op-doc)
add_circt_doc(HWAttributes Dialects/HWAttributes -gen-attrdef-doc)
add_circt_doc(HWAttributesNaming Dialects/HWAttributesNaming -gen-attrdef-doc)
add_circt_doc(HWMiscOps Dialects/HWMiscOps -gen-op-doc)
add_circt_doc(HWOpInterfaces Dialects/HWOpInterfaces -gen-op-interface-docs)
add_circt_doc(HWStructure Dialects/HWStructureOps -gen-op-doc)
add_circt_doc(HWTypeDecls Dialects/HWTypeDeclsOps -gen-op-doc)
add_circt_doc(HWTypeInterfaces Dialects/HWTypeInterfaces -gen-type-interface-docs)
add_circt_doc(HWTypes Dialects/HWTypes -gen-typedef-doc)
add_circt_doc(HWTypesImpl Dialects/HWTypesImpl -gen-typedef-doc)

# Generate Pass documentation.
add_circt_doc(Passes HWPasses -gen-pass-doc)
