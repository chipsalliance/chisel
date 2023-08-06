//===-- circt-c/Dialect/MSFT.h - C API for MSFT dialect -----------*- C -*-===//
//
// This header declares the C interface for registering and accessing the
// MSFT dialect. A dialect should be registered with a context to make it
// available to users of the context. These users must load the dialect
// before using any of its attributes, operations or types. Parser and pass
// manager can load registered dialects automatically.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_DIALECT_MSFT_H
#define CIRCT_C_DIALECT_MSFT_H

#include "circt/Dialect/MSFT/MSFTDialect.h"
#include "mlir-c/IR.h"
#include "mlir-c/Pass.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(MSFT, msft);

MLIR_CAPI_EXPORTED void mlirMSFTRegisterPasses();

// Values represented in `MSFT.td`.
typedef int32_t CirctMSFTPrimitiveType;

// Replace all uses of Value with new value.
MLIR_CAPI_EXPORTED void circtMSFTReplaceAllUsesWith(MlirValue value,
                                                    MlirValue newValue);

//===----------------------------------------------------------------------===//
// MSFT Attributes.
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool
    circtMSFTAttributeIsAPhysLocationAttribute(MlirAttribute);
MLIR_CAPI_EXPORTED MlirAttribute circtMSFTPhysLocationAttrGet(
    MlirContext, CirctMSFTPrimitiveType, uint64_t x, uint64_t y, uint64_t num);
MLIR_CAPI_EXPORTED CirctMSFTPrimitiveType
    circtMSFTPhysLocationAttrGetPrimitiveType(MlirAttribute);
MLIR_CAPI_EXPORTED uint64_t circtMSFTPhysLocationAttrGetX(MlirAttribute);
MLIR_CAPI_EXPORTED uint64_t circtMSFTPhysLocationAttrGetY(MlirAttribute);
MLIR_CAPI_EXPORTED uint64_t circtMSFTPhysLocationAttrGetNum(MlirAttribute);

MLIR_CAPI_EXPORTED MlirOperation circtMSFTGetInstance(MlirOperation root,
                                                      MlirAttribute path);

MLIR_CAPI_EXPORTED bool circtMSFTAttributeIsAPhysicalBoundsAttr(MlirAttribute);
MLIR_CAPI_EXPORTED
MlirAttribute circtMSFTPhysicalBoundsAttrGet(MlirContext, uint64_t, uint64_t,
                                             uint64_t, uint64_t);

MLIR_CAPI_EXPORTED bool
    circtMSFTAttributeIsALocationVectorAttribute(MlirAttribute);
MLIR_CAPI_EXPORTED MlirAttribute
circtMSFTLocationVectorAttrGet(MlirContext, MlirType type, intptr_t numElements,
                               MlirAttribute const *elements);
MLIR_CAPI_EXPORTED MlirType circtMSFTLocationVectorAttrGetType(MlirAttribute);
MLIR_CAPI_EXPORTED
intptr_t circtMSFTLocationVectorAttrGetNumElements(MlirAttribute);
MLIR_CAPI_EXPORTED MlirAttribute
circtMSFTLocationVectorAttrGetElement(MlirAttribute attr, intptr_t pos);

MLIR_CAPI_EXPORTED bool circtMSFTAttributeIsAnAppIDAttr(MlirAttribute);
MLIR_CAPI_EXPORTED
MlirAttribute circtMSFTAppIDAttrGet(MlirContext, MlirStringRef name,
                                    uint64_t index);
MLIR_CAPI_EXPORTED MlirStringRef circtMSFTAppIDAttrGetName(MlirAttribute attr);
MLIR_CAPI_EXPORTED uint64_t circtMSFTAppIDAttrGetIndex(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// PrimitiveDB.
//===----------------------------------------------------------------------===//

typedef struct {
  void *ptr;
} CirctMSFTPrimitiveDB;

MLIR_CAPI_EXPORTED CirctMSFTPrimitiveDB circtMSFTCreatePrimitiveDB(MlirContext);
MLIR_CAPI_EXPORTED void circtMSFTDeletePrimitiveDB(CirctMSFTPrimitiveDB self);
MLIR_CAPI_EXPORTED MlirLogicalResult circtMSFTPrimitiveDBAddPrimitive(
    CirctMSFTPrimitiveDB, MlirAttribute locAndPrim);
MLIR_CAPI_EXPORTED bool
circtMSFTPrimitiveDBIsValidLocation(CirctMSFTPrimitiveDB,
                                    MlirAttribute locAndPrim);

//===----------------------------------------------------------------------===//
// PlacementDB.
//===----------------------------------------------------------------------===//

typedef struct {
  void *ptr;
} CirctMSFTPlacementDB;

enum CirctMSFTDirection { NONE = 0, ASC = 1, DESC = 2 };
typedef struct {
  CirctMSFTDirection columns;
  CirctMSFTDirection rows;
} CirctMSFTWalkOrder;

MLIR_CAPI_EXPORTED CirctMSFTPlacementDB
circtMSFTCreatePlacementDB(MlirModule top, CirctMSFTPrimitiveDB seed);
MLIR_CAPI_EXPORTED void circtMSFTDeletePlacementDB(CirctMSFTPlacementDB self);
MLIR_CAPI_EXPORTED
MLIR_CAPI_EXPORTED MlirOperation circtMSFTPlacementDBPlace(
    CirctMSFTPlacementDB, MlirOperation inst, MlirAttribute loc,
    MlirStringRef subpath, MlirLocation srcLoc);
MLIR_CAPI_EXPORTED void
circtMSFTPlacementDBRemovePlacement(CirctMSFTPlacementDB, MlirOperation locOp);
MLIR_CAPI_EXPORTED MlirLogicalResult circtMSFTPlacementDBMovePlacement(
    CirctMSFTPlacementDB, MlirOperation locOp, MlirAttribute newLoc);
MLIR_CAPI_EXPORTED MlirOperation
circtMSFTPlacementDBGetInstanceAt(CirctMSFTPlacementDB, MlirAttribute loc);
MLIR_CAPI_EXPORTED MlirAttribute circtMSFTPlacementDBGetNearestFreeInColumn(
    CirctMSFTPlacementDB, CirctMSFTPrimitiveType prim, uint64_t column,
    uint64_t nearestToY);

typedef void (*CirctMSFTPlacementCallback)(MlirAttribute loc,
                                           MlirOperation locOp, void *userData);
/// Walk all the placements within 'bounds' ([xmin, xmax, ymin, ymax], inclusive
/// on all sides), with -1 meaning unbounded.
MLIR_CAPI_EXPORTED void circtMSFTPlacementDBWalkPlacements(
    CirctMSFTPlacementDB, CirctMSFTPlacementCallback, int64_t bounds[4],
    CirctMSFTPrimitiveType primTypeFilter, CirctMSFTWalkOrder walkOrder,
    void *userData);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_DIALECT_MSFT_H
