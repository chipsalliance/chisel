add_circt_dialect(FSM fsm)
add_circt_dialect_doc(FSM fsm)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(CIRCTFSMTransformsIncGen)
add_circt_doc(Passes FSMPasses -gen-pass-doc)
