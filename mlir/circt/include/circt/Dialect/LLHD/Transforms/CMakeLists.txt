set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(CIRCTLLHDTransformsIncGen)

add_circt_doc(Passes LLHDPasses -gen-pass-doc)
