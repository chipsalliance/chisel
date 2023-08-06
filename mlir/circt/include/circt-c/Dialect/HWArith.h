//===-- circt-c/Dialect/HWArith.h - C API for HWArith dialect -----*- C -*-===//
//
// This header declares the C interface for registering and accessing the
// HWArith dialect. A dialect should be registered with a context to make it
// available to users of the context. These users must load the dialect
// before using any of its attributes, operations or types. Parser and pass
// manager can load registered dialects automatically.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_DIALECT_HWArith_H
#define CIRCT_C_DIALECT_HWArith_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(HWArith, hwarith);
MLIR_CAPI_EXPORTED void registerHWArithPasses();

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_DIALECT_HWArith_H
