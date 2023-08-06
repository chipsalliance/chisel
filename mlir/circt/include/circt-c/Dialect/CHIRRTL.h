//===-- circt-c/Dialect/CHIRRTL.h - C API for CHIRRTL dialect -----*- C -*-===//
//
// This header declares the C interface for registering and accessing the
// CHIRRTL dialect. A dialect should be registered with a context to make it
// available to users of the context. These users must load the dialect
// before using any of its attributes, operations or types. Parser and pass
// manager can load registered dialects automatically.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_DIALECT_CHIRRTL_H
#define CIRCT_C_DIALECT_CHIRRTL_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Dialect API.
//===----------------------------------------------------------------------===//

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(CHIRRTL, chirrtl);

//===----------------------------------------------------------------------===//
// Type API.
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirType chirrtlTypeGetCMemory(MlirContext ctx,
                                                  MlirType elementType,
                                                  uint64_t numElements);

MLIR_CAPI_EXPORTED MlirType chirrtlTypeGetCMemoryPort(MlirContext ctx);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_DIALECT_CHIRRTL_H
