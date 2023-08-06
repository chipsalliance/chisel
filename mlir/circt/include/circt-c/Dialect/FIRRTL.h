//===-- circt-c/Dialect/FIRRTL.h - C API for FIRRTL dialect -------*- C -*-===//
//
// This header declares the C interface for registering and accessing the
// FIRRTL dialect. A dialect should be registered with a context to make it
// available to users of the context. These users must load the dialect
// before using any of its attributes, operations or types. Parser and pass
// manager can load registered dialects automatically.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_DIALECT_FIRRTL_H
#define CIRCT_C_DIALECT_FIRRTL_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

// NOLINTNEXTLINE(modernize-use-using)
typedef struct FIRRTLBundleField {
  MlirIdentifier name;
  bool isFlip;
  MlirType type;
} FIRRTLBundleField;

// NOLINTNEXTLINE(modernize-use-using)
typedef enum FIRRTLConvention {
  FIRRTL_CONVENTION_INTERNAL,
  FIRRTL_CONVENTION_SCALARIZED,
} FIRRTLConvention;

// NOLINTNEXTLINE(modernize-use-using)
typedef enum FIRRTLPortDir {
  FIRRTL_PORT_DIR_INPUT,
  FIRRTL_PORT_DIR_OUTPUT,
} FIRRTLPortDir;

// NOLINTNEXTLINE(modernize-use-using)
typedef enum FIRRTLNameKind {
  FIRRTL_NAME_KIND_DROPPABLE_NAME,
  FIRRTL_NAME_KIND_INTERESTING_NAME,
} FIRRTLNameKind;

// NOLINTNEXTLINE(modernize-use-using)
typedef enum FIRRTLRUW {
  FIRRTL_RUW_UNDEFINED,
  FIRRTL_RUW_OLD,
  FIRRTL_RUW_NEW,
} FIRRTLRUW;

// NOLINTNEXTLINE(modernize-use-using)
typedef enum FIRRTLMemDir {
  FIRRTL_MEM_DIR_INFER,
  FIRRTL_MEM_DIR_READ,
  FIRRTL_MEM_DIR_WRITE,
  FIRRTL_MEM_DIR_READ_WRITE,
} FIRRTLMemDir;

// NOLINTNEXTLINE(modernize-use-using)
typedef enum FIRRTLEventControl {
  FIRRTL_EVENT_CONTROL_AT_POS_EDGE,
  FIRRTL_EVENT_CONTROL_AT_NEG_EDGE,
  FIRRTL_EVENT_CONTROL_AT_EDGE,
} FIRRTLEventControl;

//===----------------------------------------------------------------------===//
// Dialect API.
//===----------------------------------------------------------------------===//

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(FIRRTL, firrtl);

//===----------------------------------------------------------------------===//
// Type API.
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirType firrtlTypeGetUInt(MlirContext ctx, int32_t width);
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetSInt(MlirContext ctx, int32_t width);
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetClock(MlirContext ctx);
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetReset(MlirContext ctx);
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetAsyncReset(MlirContext ctx);
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetAnalog(MlirContext ctx, int32_t width);
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetVector(MlirContext ctx,
                                                MlirType element, size_t count);
MLIR_CAPI_EXPORTED MlirType firrtlTypeGetBundle(
    MlirContext ctx, size_t count, const FIRRTLBundleField *fields);

//===----------------------------------------------------------------------===//
// Attribute API.
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute
firrtlAttrGetConvention(MlirContext ctx, FIRRTLConvention convention);

MLIR_CAPI_EXPORTED MlirAttribute
firrtlAttrGetPortDirs(MlirContext ctx, size_t count, const FIRRTLPortDir *dirs);

MLIR_CAPI_EXPORTED MlirAttribute firrtlAttrGetNameKind(MlirContext ctx,
                                                       FIRRTLNameKind nameKind);

MLIR_CAPI_EXPORTED MlirAttribute firrtlAttrGetRUW(MlirContext ctx,
                                                  FIRRTLRUW ruw);

MLIR_CAPI_EXPORTED MlirAttribute firrtlAttrGetMemInit(MlirContext ctx,
                                                      MlirIdentifier filename,
                                                      bool isBinary,
                                                      bool isInline);

MLIR_CAPI_EXPORTED MlirAttribute firrtlAttrGetMemDir(MlirContext ctx,
                                                     FIRRTLMemDir dir);

MLIR_CAPI_EXPORTED MlirAttribute
firrtlAttrGetEventControl(MlirContext ctx, FIRRTLEventControl eventControl);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_DIALECT_FIRRTL_H
