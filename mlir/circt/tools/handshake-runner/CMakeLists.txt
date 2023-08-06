add_llvm_executable(handshake-runner handshake-runner.cpp Simulation.cpp)

llvm_update_compile_flags(handshake-runner)
target_link_libraries(handshake-runner PRIVATE
  CIRCTHandshake
  CIRCTStandardToHandshake
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRIR
  MLIRParser
  MLIRSupport
)
