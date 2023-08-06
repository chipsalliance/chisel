//===- DeviceDB.cpp - Implement a device database -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/MSFT/DeviceDB.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/MSFT/MSFTOps.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace msft;

//===----------------------------------------------------------------------===//
// PrimitiveDB.
//===----------------------------------------------------------------------===//
// NOTE: Nothing in this implementation is in any way the most optimal
// implementation. We put off deciding what the correct data structure is until
// we have a better handle of the operations it must accelerate. Performance is
// not an immediate goal.
//===----------------------------------------------------------------------===//

PrimitiveDB::PrimitiveDB(MLIRContext *ctxt) : ctxt(ctxt) {}

/// Assign an instance to a primitive. Return false if another instance is
/// already placed at that location.
LogicalResult PrimitiveDB::addPrimitive(PhysLocationAttr loc) {
  DenseSet<PrimitiveType> &primsAtLoc = getLeaf(loc);
  PrimitiveType prim = loc.getPrimitiveType().getValue();
  if (primsAtLoc.contains(prim))
    return failure();
  primsAtLoc.insert(prim);
  return success();
}

/// Assign an instance to a primitive. Return false if another instance is
/// already placed at that location.
/// Check to see if a primitive exists.
bool PrimitiveDB::isValidLocation(PhysLocationAttr loc) {
  DenseSet<PrimitiveType> primsAtLoc = getLeaf(loc);
  return primsAtLoc.contains(loc.getPrimitiveType().getValue());
}

PrimitiveDB::DimPrimitiveType &PrimitiveDB::getLeaf(PhysLocationAttr loc) {
  return placements[loc.getX()][loc.getY()][loc.getNum()];
}

void PrimitiveDB::foreach (
    function_ref<void(PhysLocationAttr)> callback) const {
  for (const auto &x : placements)
    for (const auto &y : x.second)
      for (const auto &n : y.second)
        for (auto p : n.second)
          callback(PhysLocationAttr::get(ctxt, PrimitiveTypeAttr::get(ctxt, p),
                                         x.first, y.first, n.first));
}

//===----------------------------------------------------------------------===//
// PlacementDB.
//===----------------------------------------------------------------------===//
// NOTE: Nothing in this implementation is in any way the most optimal
// implementation. We put off deciding what the correct data structure is until
// we have a better handle of the operations it must accelerate. Performance is
// not an immediate goal.
//===----------------------------------------------------------------------===//

PlacementDB::PlacementDB(mlir::ModuleOp topMod)
    : ctxt(topMod->getContext()), topMod(topMod), seeded(false) {
  addDesignPlacements();
}
PlacementDB::PlacementDB(mlir::ModuleOp topMod, const PrimitiveDB &seed)
    : ctxt(topMod->getContext()), topMod(topMod), seeded(false) {

  seed.foreach ([this](PhysLocationAttr loc) { (void)getLeaf(loc); });
  seeded = true;
  addDesignPlacements();
}

/// Assign an instance to a primitive. Return null if another instance is
/// already placed at that location
PDPhysLocationOp PlacementDB::place(DynamicInstanceOp inst,
                                    PhysLocationAttr loc, StringRef subPath,
                                    Location srcLoc) {
  StringAttr subPathAttr;
  if (!subPath.empty())
    subPathAttr = StringAttr::get(inst->getContext(), subPath);
  PDPhysLocationOp locOp =
      OpBuilder(inst.getBody())
          .create<PDPhysLocationOp>(srcLoc, loc, subPathAttr,
                                    FlatSymbolRefAttr());
  if (succeeded(insertPlacement(locOp, locOp.getLoc())))
    return locOp;
  locOp->erase();
  return {};
}
PDRegPhysLocationOp PlacementDB::place(DynamicInstanceOp inst,
                                       LocationVectorAttr locs,
                                       Location srcLoc) {
  PDRegPhysLocationOp locOp =
      OpBuilder(inst.getBody())
          .create<PDRegPhysLocationOp>(srcLoc, locs, FlatSymbolRefAttr());
  for (PhysLocationAttr loc : locs.getLocs())
    if (failed(insertPlacement(locOp, loc))) {
      locOp->erase();
      return {};
    }
  return locOp;
}

LogicalResult PlacementDB::insertPlacement(DynInstDataOpInterface op,
                                           PhysLocationAttr loc) {
  if (!loc)
    return success();
  PlacementCell *leaf = getLeaf(loc);
  if (!leaf)
    return op->emitOpError("Could not apply placement. Invalid location: ")
           << loc;
  if (leaf->locOp != nullptr)
    return op->emitOpError("Could not apply placement ")
           << loc << ". Position already occupied by "
           << cast<DynamicInstanceOp>(leaf->locOp->getParentOp())
                  .globalRefPath();

  leaf->locOp = op;
  return success();
}

/// Assign an operation to a physical region. Return false on failure.
PDPhysRegionOp PlacementDB::placeIn(DynamicInstanceOp inst,
                                    DeclPhysicalRegionOp physregion,
                                    StringRef subPath, Location srcLoc) {
  StringAttr subPathAttr;
  if (!subPath.empty())
    subPathAttr = StringAttr::get(inst->getContext(), subPath);
  PDPhysRegionOp regOp =
      OpBuilder::atBlockEnd(&inst.getBody().front())
          .create<PDPhysRegionOp>(srcLoc, FlatSymbolRefAttr::get(physregion),
                                  subPathAttr, FlatSymbolRefAttr());
  regionPlacements.push_back(regOp);
  return regOp;
}

/// Using the operation attributes, add the proper placements to the database.
/// Return the number of placements which weren't added due to conflicts.
size_t PlacementDB::addPlacements(DynamicInstanceOp inst) {
  size_t numFailed = 0;
  inst->walk([&](Operation *op) {
    LogicalResult added = TypeSwitch<Operation *, LogicalResult>(op)
                              .Case([&](PDPhysLocationOp op) {
                                return insertPlacement(op, op.getLoc());
                              })
                              .Case([&](PDRegPhysLocationOp op) {
                                ArrayRef<PhysLocationAttr> locs =
                                    op.getLocs().getLocs();
                                for (auto loc : locs)
                                  if (failed(insertPlacement(op, loc)))
                                    return failure();
                                return success();
                              })
                              .Case([&](PDPhysRegionOp op) {
                                regionPlacements.push_back(op);
                                return success();
                              })
                              .Default([](Operation *op) { return failure(); });
    if (failed(added))
      ++numFailed;
  });

  return numFailed;
}

/// Walk the entire design adding placements.
size_t PlacementDB::addDesignPlacements() {
  size_t failed = 0;
  for (auto inst : topMod.getOps<DynamicInstanceOp>())
    failed += addPlacements(inst);
  return failed;
}

/// Remove the placement at a given location. Returns failure if nothing was
/// placed there.
void PlacementDB::removePlacement(PDPhysLocationOp locOp) {
  removePlacement(locOp, locOp.getLoc());
  locOp.erase();
}

/// Move the placement at a given location to a new location. Returns failure
/// if nothing was placed at the previous location or something is already
/// placed at the new location.
LogicalResult PlacementDB::movePlacement(PDPhysLocationOp locOp,
                                         PhysLocationAttr newLoc) {
  PhysLocationAttr from = locOp.getLoc();
  if (failed(movePlacementCheck(locOp, from, newLoc)))
    return failure();
  locOp.setLocAttr(newLoc);
  movePlacement(locOp, from, newLoc);
  return success();
}

/// Remove the placement at a given location. Returns failure if nothing was
/// placed there.
void PlacementDB::removePlacement(PDRegPhysLocationOp locOp) {
  for (PhysLocationAttr loc : locOp.getLocs().getLocs())
    if (loc)
      removePlacement(locOp, loc);
  locOp.erase();
}

/// Move the placement at a given location to a new location. Returns failure
/// if nothing was placed at the previous location or something is already
/// placed at the new location.
LogicalResult PlacementDB::movePlacement(PDRegPhysLocationOp locOp,
                                         LocationVectorAttr newLocs) {
  ArrayRef<PhysLocationAttr> fromLocs = locOp.getLocs().getLocs();

  // Check that each move/insert/delete will succeed before doing any of the
  // mutations.
  for (auto [from, to] : llvm::zip(fromLocs, newLocs.getLocs())) {
    // If 'from' and 'to' are both non-null, this location is being moved.
    if (from && to && failed(movePlacementCheck(locOp, from, to)))
      return failure();
    // If only 'to' is valid, this location is the initial placement.
    if (to && getInstanceAt(to))
      return failure();
    // If 'to' isn't valid, it's a placement removal which will always succeed.
  }

  // Mutate.
  for (auto [from, to] : llvm::zip(fromLocs, newLocs.getLocs())) {
    // If 'from' and 'to' are both non-null, this location is being moved.
    if (from && to)
      movePlacement(locOp, from, to);
    // If 'to' isn't valid, it's a placement removal.
    else if (from)
      removePlacement(locOp, from);
    // If only 'to' is valid, this location is the initial placement. Since we
    // checked that there isn't anything currently located at 'to' this call
    // will never fail.
    else if (to)
      (void)insertPlacement(locOp, to);
  }

  locOp.setLocsAttr(newLocs);
  return success();
}

void PlacementDB::removePlacement(DynInstDataOpInterface op,
                                  PhysLocationAttr loc) {
  PlacementCell *leaf = getLeaf(loc);
  assert(leaf && "Could not find op at location specified by op");
  assert(leaf->locOp == op);
  leaf->locOp = {};
}

LogicalResult PlacementDB::movePlacementCheck(DynInstDataOpInterface op,
                                              PhysLocationAttr from,
                                              PhysLocationAttr to) {
  if (from == to)
    return success();

  PlacementCell *oldLeaf = getLeaf(from);
  PlacementCell *newLeaf = getLeaf(to);

  if (!oldLeaf || !newLeaf)
    return failure();

  if (oldLeaf->locOp == nullptr)
    return op.emitError("cannot move from a location not occupied by "
                        "specified op. Currently unoccupied");
  if (oldLeaf->locOp != op)
    return op.emitError("cannot move from a location not occupied by "
                        "specified op. Currently occupied by ")
           << oldLeaf->locOp;
  if (newLeaf->locOp)
    return op.emitError(
               "cannot move to new location since location is occupied by ")
           << cast<DynamicInstanceOp>(newLeaf->locOp->getParentOp())
                  .globalRefPath();
  return success();
}

void PlacementDB::movePlacement(DynInstDataOpInterface op,
                                PhysLocationAttr from, PhysLocationAttr to) {
  assert(succeeded(movePlacementCheck(op, from, to)) &&
         "Call `movePlacementCheck` first to ensure that move is legal.");
  if (from == to)
    return;
  PlacementCell *oldLeaf = getLeaf(from);
  PlacementCell *newLeaf = getLeaf(to);
  newLeaf->locOp = op;
  oldLeaf->locOp = {};
}

/// Lookup the instance at a particular location.
DynInstDataOpInterface PlacementDB::getInstanceAt(PhysLocationAttr loc) {
  auto innerMap = placements[loc.getX()][loc.getY()][loc.getNum()];
  auto instF = innerMap.find(loc.getPrimitiveType().getValue());
  if (instF == innerMap.end())
    return {};
  if (!instF->getSecond().locOp)
    return {};
  return instF->getSecond().locOp;
}

PhysLocationAttr PlacementDB::getNearestFreeInColumn(PrimitiveType prim,
                                                     uint64_t columnNum,
                                                     uint64_t nearestToY) {
  // Simplest possible algorithm.
  PhysLocationAttr nearest = {};
  walkPlacements(
      [&nearest, nearestToY](PhysLocationAttr loc, Operation *locOp) {
        if (locOp)
          return;
        if (!nearest) {
          nearest = loc;
          return;
        }
        int64_t curDist =
            std::abs((int64_t)nearestToY - (int64_t)nearest.getY());
        int64_t replDist = std::abs((int64_t)nearestToY - (int64_t)loc.getY());
        if (replDist < curDist)
          nearest = loc;
      },
      std::make_tuple(columnNum, columnNum, -1, -1), prim);
  return nearest;
}

PlacementDB::PlacementCell *PlacementDB::getLeaf(PhysLocationAttr loc) {
  PrimitiveType primType = loc.getPrimitiveType().getValue();

  DimNumMap &nums = placements[loc.getX()][loc.getY()];
  if (!seeded)
    return &nums[loc.getNum()][primType];
  if (!nums.count(loc.getNum()))
    return {};

  DimDevType &primitives = nums[loc.getNum()];
  if (primitives.count(primType) == 0)
    return {};
  return &primitives[primType];
}

/// Walker for placements.
void PlacementDB::walkPlacements(
    function_ref<void(PhysLocationAttr, DynInstDataOpInterface)> callback,
    std::tuple<int64_t, int64_t, int64_t, int64_t> bounds,
    std::optional<PrimitiveType> primType, std::optional<WalkOrder> walkOrder) {
  uint64_t xmin = std::get<0>(bounds) < 0 ? 0 : std::get<0>(bounds);
  uint64_t xmax = std::get<1>(bounds) < 0 ? std::numeric_limits<uint64_t>::max()
                                          : (uint64_t)std::get<1>(bounds);
  uint64_t ymin = std::get<2>(bounds) < 0 ? 0 : std::get<2>(bounds);
  uint64_t ymax = std::get<3>(bounds) < 0 ? std::numeric_limits<uint64_t>::max()
                                          : (uint64_t)std::get<3>(bounds);

  // TODO: Since the data structures we're using aren't sorted, the best we can
  // do is iterate and filter. If a specific order is requested, we sort the
  // keys by that as we go. Once we get to performance, we'll figure out the
  // right data structure.

  auto maybeSort = [](auto &container, auto direction) {
    if (!direction.has_value())
      return;
    if (*direction == Direction::NONE)
      return;

    llvm::sort(container, [direction](auto colA, auto colB) {
      if (*direction == Direction::ASC)
        return colA.first < colB.first;

      return colA.first > colB.first;
    });
  };

  // X loop.
  SmallVector<std::pair<size_t, DimYMap>> cols(placements.begin(),
                                               placements.end());
  maybeSort(cols, llvm::transformOptional(walkOrder,
                                          [](auto wo) { return wo.columns; }));
  for (const auto &colF : cols) {
    size_t x = colF.first;
    if (x < xmin || x > xmax)
      continue;
    DimYMap yMap = colF.second;

    // Y loop.
    SmallVector<std::pair<size_t, DimNumMap>> rows(yMap.begin(), yMap.end());
    maybeSort(rows, llvm::transformOptional(walkOrder,
                                            [](auto wo) { return wo.rows; }));
    for (const auto &rowF : rows) {
      size_t y = rowF.first;
      if (y < ymin || y > ymax)
        continue;
      DimNumMap numMap = rowF.second;

      // Num loop.
      for (auto &numF : numMap) {
        size_t num = numF.getFirst();
        DimDevType devMap = numF.getSecond();

        // DevType loop.
        for (auto &devF : devMap) {
          PrimitiveType devtype = devF.getFirst();
          if (primType && devtype != *primType)
            continue;
          PlacementCell &inst = devF.getSecond();

          // Marshall and run the callback.
          PhysLocationAttr loc = PhysLocationAttr::get(
              ctxt, PrimitiveTypeAttr::get(ctxt, devtype), x, y, num);
          callback(loc, inst.locOp);
        }
      }
    }
  }
}

/// Walk the region placement information.
void PlacementDB::walkRegionPlacements(
    function_ref<void(PDPhysRegionOp)> callback) {
  for (auto regOp : regionPlacements)
    callback(regOp);
}
