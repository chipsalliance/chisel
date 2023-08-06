add_llvm_executable(mlir-bindings-support-test
  NativeReferenceTest.cpp)
target_link_libraries(mlir-bindings-support-test PRIVATE
  MLIRBindingsSupport)
