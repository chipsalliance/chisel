set(LLVM_OPTIONAL_SOURCES
  DialectRegistration.cpp
  mlir_std_runner.cpp
  Ops.cpp
  )

add_circt_dialect_library(CIRCTPipelineOps
  PipelineOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include

  LINK_LIBS PUBLIC
  CIRCTESI
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR

  DEPENDS
  MLIRArithDialect
  MLIRPipelineIncGen
)

add_subdirectory(Transforms)
