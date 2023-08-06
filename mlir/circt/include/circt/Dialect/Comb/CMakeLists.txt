add_circt_dialect(Comb comb)
add_circt_dialect_doc(Comb comb)

set(LLVM_TARGET_DEFINITIONS Comb.td)
mlir_tablegen(CombEnums.h.inc -gen-enum-decls)
mlir_tablegen(CombEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRCombEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(CIRCTCombTransformsIncGen)

# Generate Pass documentation.
add_circt_doc(Passes CombPasses -gen-pass-doc)
