# FIRRTL
add_circt_dialect(FIRRTL firrtl FIRRTL)

set(LLVM_TARGET_DEFINITIONS FIRRTLEnums.td)
mlir_tablegen(FIRRTLEnums.h.inc -gen-enum-decls)
mlir_tablegen(FIRRTLEnums.cpp.inc -gen-enum-defs)

set(LLVM_TARGET_DEFINITIONS FIRRTL.td)
mlir_tablegen(FIRRTLAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=firrtl)
mlir_tablegen(FIRRTLAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=firrtl)
add_public_tablegen_target(CIRCTFIRRTLEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(CIRCTFIRRTLTransformsIncGen)

set(LLVM_TARGET_DEFINITIONS FIRRTLCanonicalization.td)
mlir_tablegen(FIRRTLCanonicalization.h.inc -gen-rewriters)
add_public_tablegen_target(CIRCTFIRRTLCanonicalizationIncGen)

set(LLVM_TARGET_DEFINITIONS FIRRTLOpInterfaces.td)
mlir_tablegen(FIRRTLOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(FIRRTLOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(CIRCTFIRRTLOpInterfacesIncGen)
add_dependencies(circt-headers CIRCTFIRRTLOpInterfacesIncGen)

# Generate Dialect documentation.
add_circt_doc(FIRRTLAttributes Dialects/FIRRTLAttributes -gen-attrdef-doc)
add_circt_doc(FIRRTLStructure Dialects/FIRRTLStructureOps -gen-op-doc)
add_circt_doc(FIRRTLDeclarations Dialects/FIRRTLDeclarationOps -gen-op-doc)
add_circt_doc(FIRRTLStatements Dialects/FIRRTLStatementOps -gen-op-doc)
add_circt_doc(FIRRTLExpressions Dialects/FIRRTLExpressionOps -gen-op-doc)
add_circt_doc(FIRRTLTypes Dialects/FIRRTLTypes -gen-typedef-doc)
add_circt_doc(FIRRTLTypesImpl Dialects/FIRRTLTypesImpl -gen-typedef-doc)
add_circt_doc(FIRRTLOpInterfaces Dialects/FIRRTLOpInterfaces -gen-op-interface-docs)

# Generate Pass documentation.
add_circt_doc(Passes FIRRTLPasses -gen-pass-doc)

# CHIRRTL
add_circt_dialect(CHIRRTL chirrtl CHIRRTL)
add_circt_dialect_doc(CHIRRTL chirrtl)
