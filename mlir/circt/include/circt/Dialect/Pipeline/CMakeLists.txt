add_circt_dialect(Pipeline pipeline)
add_circt_dialect_doc(Pipeline pipeline)

set(LLVM_TARGET_DEFINITIONS Pipeline.td)

set(LLVM_TARGET_DEFINITIONS PipelinePasses.td)
mlir_tablegen(PipelinePasses.h.inc -gen-pass-decls)
add_public_tablegen_target(CIRCTPipelineTransformsIncGen)
add_circt_doc(PipelinePasses PipelinePasses -gen-pass-doc)
