//===- circt-c/ExportFIRRTL.h - C API for emitting FIRRTL ---------*- C -*-===//
//
// This header declares the C interface for emitting FIRRTL from a CIRCT MLIR
// module.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_EXPORTFIRRTL_H
#define CIRCT_C_EXPORTFIRRTL_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Emits FIRRTL for the specified module using the provided callback and user
/// data
MLIR_CAPI_EXPORTED MlirLogicalResult mlirExportFIRRTL(MlirModule,
                                                      MlirStringCallback,
                                                      void *userData);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_EXPORTFIRRTL_H
