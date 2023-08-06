//===- MSFT.cpp - C Interface for the MSFT Dialect ------------------------===//
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/MSFT.h"
#include "circt/Dialect/HW/HWSymCache.h"
#include "circt/Dialect/MSFT/DeviceDB.h"
#include "circt/Dialect/MSFT/ExportTcl.h"
#include "circt/Dialect/MSFT/MSFTAttributes.h"
#include "circt/Dialect/MSFT/MSFTDialect.h"
#include "circt/Dialect/MSFT/MSFTOps.h"
#include "circt/Dialect/MSFT/MSFTPasses.h"
#include "circt/Support/LLVM.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(MSFT, msft, circt::msft::MSFTDialect)

using namespace circt;
using namespace circt::msft;

void mlirMSFTRegisterPasses() {
  mlir::registerCanonicalizerPass();
  registerPasses();
}

void circtMSFTReplaceAllUsesWith(MlirValue value, MlirValue newValue) {
  unwrap(value).replaceAllUsesWith(unwrap(newValue));
}

//===----------------------------------------------------------------------===//
// PrimitiveDB.
//===----------------------------------------------------------------------===//

DEFINE_C_API_PTR_METHODS(CirctMSFTPrimitiveDB, circt::msft::PrimitiveDB)

CirctMSFTPrimitiveDB circtMSFTCreatePrimitiveDB(MlirContext ctxt) {
  return wrap(new PrimitiveDB(unwrap(ctxt)));
}
void circtMSFTDeletePrimitiveDB(CirctMSFTPrimitiveDB self) {
  delete unwrap(self);
}
MlirLogicalResult circtMSFTPrimitiveDBAddPrimitive(CirctMSFTPrimitiveDB self,
                                                   MlirAttribute cLoc) {
  PhysLocationAttr loc = unwrap(cLoc).cast<PhysLocationAttr>();
  return wrap(unwrap(self)->addPrimitive(loc));
}
bool circtMSFTPrimitiveDBIsValidLocation(CirctMSFTPrimitiveDB self,
                                         MlirAttribute cLoc) {
  PhysLocationAttr loc = unwrap(cLoc).cast<PhysLocationAttr>();
  return unwrap(self)->isValidLocation(loc);
}

//===----------------------------------------------------------------------===//
// PlacementDB.
//===----------------------------------------------------------------------===//

DEFINE_C_API_PTR_METHODS(CirctMSFTPlacementDB, circt::msft::PlacementDB)

CirctMSFTPlacementDB circtMSFTCreatePlacementDB(MlirModule top,
                                                CirctMSFTPrimitiveDB seed) {
  if (seed.ptr == nullptr)
    return wrap(new PlacementDB(unwrap(top)));
  return wrap(new PlacementDB(unwrap(top), *unwrap(seed)));
}
void circtMSFTDeletePlacementDB(CirctMSFTPlacementDB self) {
  delete unwrap(self);
}

MlirOperation circtMSFTPlacementDBPlace(CirctMSFTPlacementDB db,
                                        MlirOperation cinst, MlirAttribute cloc,
                                        MlirStringRef subpath,
                                        MlirLocation csrcLoc) {
  auto inst = cast<DynamicInstanceOp>(unwrap(cinst));
  Location srcLoc = unwrap(csrcLoc);
  Attribute locAttr = unwrap(cloc);

  if (auto pla = locAttr.dyn_cast<PhysLocationAttr>())
    return wrap(unwrap(db)->place(inst, pla, unwrap(subpath), srcLoc));
  if (auto locVec = locAttr.dyn_cast<LocationVectorAttr>())
    return wrap(unwrap(db)->place(inst, locVec, srcLoc));
  llvm_unreachable("Can only place PDPhysLocationOp and PDRegPhysLocationOp");
}
void circtMSFTPlacementDBRemovePlacement(CirctMSFTPlacementDB db,
                                         MlirOperation clocOp) {
  Operation *locOp = unwrap(clocOp);
  if (auto physLocOp = dyn_cast<PDPhysLocationOp>(locOp))
    unwrap(db)->removePlacement(physLocOp);
  else if (auto regPhysLocOp = dyn_cast<PDRegPhysLocationOp>(locOp))
    unwrap(db)->removePlacement(regPhysLocOp);
  else
    assert(false && "Can only remove PDPhysLocationOp and PDRegPhysLocationOp");
}
MlirLogicalResult circtMSFTPlacementDBMovePlacement(CirctMSFTPlacementDB db,
                                                    MlirOperation clocOp,
                                                    MlirAttribute cnewLoc) {
  Operation *locOp = unwrap(clocOp);
  Attribute newLoc = unwrap(cnewLoc);
  if (auto physLocOp = dyn_cast<PDPhysLocationOp>(locOp))
    return wrap(
        unwrap(db)->movePlacement(physLocOp, newLoc.cast<PhysLocationAttr>()));
  if (auto regPhysLocOp = dyn_cast<PDRegPhysLocationOp>(locOp))
    return wrap(unwrap(db)->movePlacement(regPhysLocOp,
                                          newLoc.cast<LocationVectorAttr>()));
  llvm_unreachable("Can only move PDPhysLocationOp and PDRegPhysLocationOp");
}
MlirOperation circtMSFTPlacementDBGetInstanceAt(CirctMSFTPlacementDB db,
                                                MlirAttribute loc) {
  return wrap(unwrap(db)->getInstanceAt(unwrap(loc).cast<PhysLocationAttr>()));
}
MlirAttribute circtMSFTPlacementDBGetNearestFreeInColumn(
    CirctMSFTPlacementDB db, CirctMSFTPrimitiveType prim, uint64_t column,
    uint64_t nearestToY) {

  return wrap(unwrap(db)->getNearestFreeInColumn((PrimitiveType)prim, column,
                                                 nearestToY));
}

/// Walk all the placements within 'bounds' ([xmin, xmax, ymin, ymax], inclusive
/// on all sides), with -1 meaning unbounded.
MLIR_CAPI_EXPORTED void circtMSFTPlacementDBWalkPlacements(
    CirctMSFTPlacementDB cdb, CirctMSFTPlacementCallback ccb, int64_t bounds[4],
    CirctMSFTPrimitiveType cPrimTypeFilter, CirctMSFTWalkOrder cWalkOrder,
    void *userData) {

  PlacementDB *db = unwrap(cdb);
  auto cb = [ccb, userData](PhysLocationAttr loc,
                            DynInstDataOpInterface locOp) {
    ccb(wrap(loc), wrap(locOp), userData);
  };
  std::optional<PrimitiveType> primTypeFilter;
  if (cPrimTypeFilter >= 0)
    primTypeFilter = static_cast<PrimitiveType>(cPrimTypeFilter);

  std::optional<PlacementDB::WalkOrder> walkOrder;
  if (cWalkOrder.columns != CirctMSFTDirection::NONE ||
      cWalkOrder.rows != CirctMSFTDirection::NONE)
    walkOrder = PlacementDB::WalkOrder{
        static_cast<PlacementDB::Direction>(cWalkOrder.columns),
        static_cast<PlacementDB::Direction>(cWalkOrder.rows)};

  db->walkPlacements(
      cb, std::make_tuple(bounds[0], bounds[1], bounds[2], bounds[3]),
      primTypeFilter, walkOrder);
}

//===----------------------------------------------------------------------===//
// MSFT Attributes.
//===----------------------------------------------------------------------===//

void mlirMSFTAddPhysLocationAttr(MlirOperation cOp, const char *entityName,
                                 PrimitiveType type, long x, long y, long num) {
  Operation *op = unwrap(cOp);
  MLIRContext *ctxt = op->getContext();
  PhysLocationAttr loc = PhysLocationAttr::get(
      ctxt, PrimitiveTypeAttr::get(ctxt, type), x, y, num);
  StringAttr entity = StringAttr::get(ctxt, entityName);
  OpBuilder(op).create<PDPhysLocationOp>(op->getLoc(), loc, entity,
                                         FlatSymbolRefAttr::get(op));
  op->setAttr(entity, loc);
}

bool circtMSFTAttributeIsAPhysLocationAttribute(MlirAttribute attr) {
  return unwrap(attr).isa<PhysLocationAttr>();
}
MlirAttribute circtMSFTPhysLocationAttrGet(MlirContext cCtxt,
                                           CirctMSFTPrimitiveType devType,
                                           uint64_t x, uint64_t y,
                                           uint64_t num) {
  auto *ctxt = unwrap(cCtxt);
  return wrap(PhysLocationAttr::get(
      ctxt, PrimitiveTypeAttr::get(ctxt, (PrimitiveType)devType), x, y, num));
}

CirctMSFTPrimitiveType
circtMSFTPhysLocationAttrGetPrimitiveType(MlirAttribute attr) {
  return (CirctMSFTPrimitiveType)unwrap(attr)
      .cast<PhysLocationAttr>()
      .getPrimitiveType()
      .getValue();
}
uint64_t circtMSFTPhysLocationAttrGetX(MlirAttribute attr) {
  return (CirctMSFTPrimitiveType)unwrap(attr).cast<PhysLocationAttr>().getX();
}
uint64_t circtMSFTPhysLocationAttrGetY(MlirAttribute attr) {
  return (CirctMSFTPrimitiveType)unwrap(attr).cast<PhysLocationAttr>().getY();
}
uint64_t circtMSFTPhysLocationAttrGetNum(MlirAttribute attr) {
  return (CirctMSFTPrimitiveType)unwrap(attr).cast<PhysLocationAttr>().getNum();
}

bool circtMSFTAttributeIsAPhysicalBoundsAttr(MlirAttribute attr) {
  return unwrap(attr).isa<PhysicalBoundsAttr>();
}

MlirAttribute circtMSFTPhysicalBoundsAttrGet(MlirContext cContext,
                                             uint64_t xMin, uint64_t xMax,
                                             uint64_t yMin, uint64_t yMax) {
  auto *context = unwrap(cContext);
  return wrap(PhysicalBoundsAttr::get(context, xMin, xMax, yMin, yMax));
}

bool circtMSFTAttributeIsALocationVectorAttribute(MlirAttribute attr) {
  return unwrap(attr).isa<LocationVectorAttr>();
}
MlirAttribute circtMSFTLocationVectorAttrGet(MlirContext ctxt, MlirType type,
                                             intptr_t numElements,
                                             MlirAttribute const *elements) {
  SmallVector<PhysLocationAttr, 32> physLocs;
  for (intptr_t i = 0; i < numElements; ++i)
    if (elements[i].ptr != nullptr)
      physLocs.push_back(unwrap(elements[i]).cast<PhysLocationAttr>());
    else
      physLocs.push_back({});
  return wrap(LocationVectorAttr::get(unwrap(ctxt), TypeAttr::get(unwrap(type)),
                                      physLocs));
}
MlirType circtMSFTLocationVectorAttrGetType(MlirAttribute attr) {
  return wrap(unwrap(attr).cast<LocationVectorAttr>().getType().getValue());
}
intptr_t circtMSFTLocationVectorAttrGetNumElements(MlirAttribute attr) {
  return unwrap(attr).cast<LocationVectorAttr>().getLocs().size();
}
MlirAttribute circtMSFTLocationVectorAttrGetElement(MlirAttribute attr,
                                                    intptr_t pos) {
  return wrap(unwrap(attr).cast<LocationVectorAttr>().getLocs()[pos]);
}

bool circtMSFTAttributeIsAnAppIDAttr(MlirAttribute attr) {
  return unwrap(attr).isa<AppIDAttr>();
}

MlirAttribute circtMSFTAppIDAttrGet(MlirContext ctxt, MlirStringRef name,
                                    uint64_t index) {
  return wrap(AppIDAttr::get(
      unwrap(ctxt), StringAttr::get(unwrap(ctxt), unwrap(name)), index));
}
MlirStringRef circtMSFTAppIDAttrGetName(MlirAttribute attr) {
  return wrap(unwrap(attr).cast<AppIDAttr>().getName().getValue());
}
uint64_t circtMSFTAppIDAttrGetIndex(MlirAttribute attr) {
  return unwrap(attr).cast<AppIDAttr>().getIndex();
}
