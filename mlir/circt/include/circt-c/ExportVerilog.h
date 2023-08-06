//===-- circt-c/SVDialect.h - C API for emitting Verilog ----------*- C -*-===//
//
// This header declares the C interface for emitting Verilog from a CIRCT MLIR
// module.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_EXPORTVERILOG_H
#define CIRCT_C_EXPORTVERILOG_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Emits verilog for the specified module using the provided callback and user
/// data
MLIR_CAPI_EXPORTED MlirLogicalResult mlirExportVerilog(MlirModule,
                                                       MlirStringCallback,
                                                       void *userData);

/// Emits split Verilog files for the specified module into the given directory.
MLIR_CAPI_EXPORTED MlirLogicalResult mlirExportSplitVerilog(MlirModule,
                                                            MlirStringRef);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_EXPORTVERILOG_H
