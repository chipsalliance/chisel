add_circt_dialect(LLHD llhd)
add_circt_dialect_doc(LLHD llhd)

set(LLVM_TARGET_DEFINITIONS LLHD.td)
mlir_tablegen(LLHDEnums.h.inc -gen-enum-decls)
mlir_tablegen(LLHDEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(CIRCTLLHDEnumsIncGen)

mlir_tablegen(LLHDAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=llhd)
mlir_tablegen(LLHDAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=llhd)
add_public_tablegen_target(CIRCTLLHDAttributesIncGen)
