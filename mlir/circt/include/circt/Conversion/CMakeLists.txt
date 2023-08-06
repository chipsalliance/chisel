set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Conversion)
add_public_tablegen_target(CIRCTConversionPassIncGen)

add_circt_doc(Passes CIRCTConversionPasses -gen-pass-doc)
