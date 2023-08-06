set(CMAKE_Swift_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/swift)
enable_language(Swift)

add_executable(mlir-bindings-swift-test
  BindingsTest.swift
)
target_link_libraries(mlir-bindings-swift-test MLIRBindingsSupport)
target_compile_options(mlir-bindings-swift-test
  PRIVATE
    # -emit-module
    "SHELL:-cxx-interoperability-mode=swift-5.9"
    "SHELL:-Xfrontend -enable-experimental-move-only"
    "SHELL:-Xfrontend -enable-experimental-feature -Xfrontend MoveOnly"
    "SHELL:-Xfrontend -enable-experimental-feature -Xfrontend MoveOnlyClass"
    "SHELL:-Xfrontend -enable-experimental-feature -Xfrontend Macros"
    "SHELL:-Xfrontend -validate-tbd-against-ir=none"
    -Xcc -std=c++17)
    