//===-- circt-c/Dialect/ESI.h - C API for ESI dialect -------------*- C -*-===//
//
// This header declares the C interface for registering and accessing the
// Comb dialect. A dialect should be registered with a context to make it
// available to users of the context. These users must load the dialect
// before using any of its attributes, operations or types. Parser and pass
// manager can load registered dialects automatically.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_DIALECT_ESI_H
#define CIRCT_C_DIALECT_ESI_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(ESI, esi);
MLIR_CAPI_EXPORTED void registerESIPasses();
MLIR_CAPI_EXPORTED void registerESITranslations();

MLIR_CAPI_EXPORTED MlirLogicalResult
circtESIExportCosimSchema(MlirModule, MlirStringCallback, void *userData);

MLIR_CAPI_EXPORTED bool circtESITypeIsAChannelType(MlirType type);
MLIR_CAPI_EXPORTED MlirType circtESIChannelTypeGet(MlirType inner,
                                                   uint32_t signaling);
MLIR_CAPI_EXPORTED MlirType circtESIChannelGetInner(MlirType channelType);
MLIR_CAPI_EXPORTED uint32_t circtESIChannelGetSignaling(MlirType channelType);

MLIR_CAPI_EXPORTED bool circtESITypeIsAnAnyType(MlirType type);
MLIR_CAPI_EXPORTED MlirType circtESIAnyTypeGet(MlirContext);

MLIR_CAPI_EXPORTED bool circtESITypeIsAListType(MlirType type);
MLIR_CAPI_EXPORTED MlirType circtESIListTypeGet(MlirType inner);
MLIR_CAPI_EXPORTED MlirType
circtESIListTypeGetElementType(MlirType channelType);

MLIR_CAPI_EXPORTED MlirOperation circtESIWrapModule(MlirOperation cModOp,
                                                    long numPorts,
                                                    const MlirStringRef *ports);

MLIR_CAPI_EXPORTED void circtESIAppendMlirFile(MlirModule,
                                               MlirStringRef fileName);
MLIR_CAPI_EXPORTED MlirOperation circtESILookup(MlirModule,
                                                MlirStringRef symbol);

typedef MlirLogicalResult (*CirctESIServiceGeneratorFunc)(
    MlirOperation serviceImplementReqOp, MlirOperation declOp, void *userData);
MLIR_CAPI_EXPORTED void circtESIRegisterGlobalServiceGenerator(
    MlirStringRef impl_type, CirctESIServiceGeneratorFunc, void *userData);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_DIALECT_ESI_H
