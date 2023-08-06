//===-- circt-c/Handshake.h - C API for Handshake dialect ---------*- C -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_DIALECT_HANDSHAKE_H
#define CIRCT_C_DIALECT_HANDSHAKE_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Handshake, handshake);
MLIR_CAPI_EXPORTED void registerHandshakePasses();

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_DIALECT_HANDSHAKE_H
