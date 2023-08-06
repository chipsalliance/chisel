// RUN: mlir-bindings-support-test 2>&1 | FileCheck %s

#include "circel/Bindings/Support/NativeReference.h"

#include <cstdint>
#include <iostream>
#include <mlir/IR/Types.h>
#include <mlir/Support/TypeID.h>

using namespace mlir::bindings;

int main() {
  static int deinitializations = 0;
  struct DeinitializationTracker {
    DeinitializationTracker() {}
    ~DeinitializationTracker() { deinitializations++; }
  };

  void *opaqueReference = nullptr;
  {
    auto reference = NativeReference<DeinitializationTracker>::create();
    opaqueReference = reference.getRetainedOpaqueReference();

    // CHECK: 0
    std::cout << deinitializations << std::endl;
  }

  // CHECK: 0
  std::cout << deinitializations << std::endl;

  {
    auto reference =
        NativeReference<DeinitializationTracker>::getFromOpaqueReference(
            opaqueReference);

    reference.getValue();

    AnyNativeReference::releaseOpaqueReference(opaqueReference);

    // CHECK: 0
    std::cout << deinitializations << std::endl;
  }

  // CHECK: 1
  std::cout << deinitializations << std::endl;

  return 0;
}
