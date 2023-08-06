//===- DeviceDB.h - Device database -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Represent the placements of primitives on an FPGA.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_MSFT_DEVICEDB_H
#define CIRCT_DIALECT_MSFT_DEVICEDB_H

#include "circt/Dialect/MSFT/MSFTAttributes.h"
#include "circt/Dialect/MSFT/MSFTOps.h"

#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"

namespace circt {
namespace msft {

/// A data structure to contain locations of the primitives on the
/// device.
class PrimitiveDB {
public:
  /// Create a DB treating 'top' as the root module.
  PrimitiveDB(MLIRContext *);

  /// Place a primitive at a location.
  LogicalResult addPrimitive(PhysLocationAttr);
  /// Check to see if a primitive exists.
  bool isValidLocation(PhysLocationAttr);

  /// Iterate over all the primitive locations, executing 'callback' on each
  /// one.
  void foreach (function_ref<void(PhysLocationAttr)> callback) const;

private:
  using DimPrimitiveType = DenseSet<PrimitiveType>;
  using DimNumMap = DenseMap<size_t, DimPrimitiveType>;
  using DimYMap = DenseMap<size_t, DimNumMap>;
  using DimXMap = DenseMap<size_t, DimYMap>;

  /// Get the leaf node. Abstract this out to make it easier to change the
  /// underlying data structure.
  DimPrimitiveType &getLeaf(PhysLocationAttr);
  // TODO: Create read-only version of getLeaf.

  DimXMap placements;
  MLIRContext *ctxt;
};

/// A data structure to contain both the locations of the primitives on the
/// device and instance assignments to said primitives locations, aka
/// placements.
///
/// Holds pointers into the IR, which may become invalid as a result of IR
/// transforms. As a result, this class is intended to be short-lived -- created
/// just before loading placements and destroyed immediatetly after things are
/// placed.
class PlacementDB {
public:
  /// Create a placement db containing all the placements in 'topMod'.
  PlacementDB(mlir::ModuleOp topMod);
  PlacementDB(mlir::ModuleOp topMod, const PrimitiveDB &seed);

  /// Contains the order to iterate in each dimension for walkPlacements. The
  /// dimensions are visited with columns first, then rows, then numbers within
  /// a cell.
  enum Direction { NONE = 0, ASC = 1, DESC = 2 };
  struct WalkOrder {
    Direction columns;
    Direction rows;
  };

  /// Assign an instance to a primitive. Return null if another instance is
  /// already placed at that location.
  PDPhysLocationOp place(DynamicInstanceOp inst, PhysLocationAttr,
                         StringRef subpath, Location srcLoc);
  PDRegPhysLocationOp place(DynamicInstanceOp inst, LocationVectorAttr,
                            Location srcLoc);
  /// Assign an operation to a physical region. Return false on failure.
  PDPhysRegionOp placeIn(DynamicInstanceOp inst, DeclPhysicalRegionOp,
                         StringRef subPath, Location srcLoc);

  /// Remove the placement from the DB and IR. Erases the op.
  void removePlacement(PDPhysLocationOp);
  /// Move a placement location to a new location. Returns failure if something
  /// is already placed at the new location.
  LogicalResult movePlacement(PDPhysLocationOp, PhysLocationAttr);

  /// Remove the placement from the DB and IR. Erases the op.
  void removePlacement(PDRegPhysLocationOp);
  /// Move a placement location to a new location. Returns failure if something
  /// is already placed at the new location.
  LogicalResult movePlacement(PDRegPhysLocationOp, LocationVectorAttr);

  /// Lookup the instance at a particular location.
  DynInstDataOpInterface getInstanceAt(PhysLocationAttr);

  /// Find the nearest unoccupied primitive location to 'nearestToY' in
  /// 'column'.
  PhysLocationAttr getNearestFreeInColumn(PrimitiveType prim, uint64_t column,
                                          uint64_t nearestToY);

  /// Walk the placement information in some sort of reasonable order. Bounds
  /// restricts the walk to a rectangle of [xmin, xmax, ymin, ymax] (inclusive),
  /// with -1 meaning unbounded.
  void
  walkPlacements(function_ref<void(PhysLocationAttr, DynInstDataOpInterface)>,
                 std::tuple<int64_t, int64_t, int64_t, int64_t> bounds =
                     std::make_tuple(-1, -1, -1, -1),
                 std::optional<PrimitiveType> primType = {},
                 std::optional<WalkOrder> = {});

  /// Walk the region placement information.
  void walkRegionPlacements(function_ref<void(PDPhysRegionOp)>);

private:
  /// A memory slot. Useful to distinguish the memory location from the
  /// reference stored there.
  struct PlacementCell {
    DynInstDataOpInterface locOp;
  };

  MLIRContext *ctxt;
  mlir::ModuleOp topMod;

  using DimDevType = DenseMap<PrimitiveType, PlacementCell>;
  using DimNumMap = DenseMap<size_t, DimDevType>;
  using DimYMap = DenseMap<size_t, DimNumMap>;
  using DimXMap = DenseMap<size_t, DimYMap>;
  using RegionPlacements = SmallVector<PDPhysRegionOp>;

  /// Get the leaf node. Abstract this out to make it easier to change the
  /// underlying data structure.
  PlacementCell *getLeaf(PhysLocationAttr);

  DimXMap placements;
  RegionPlacements regionPlacements;
  bool seeded;

  /// Load the placements from `inst`.  Return the number of placements which
  /// weren't added due to conflicts.
  size_t addPlacements(DynamicInstanceOp inst);
  LogicalResult insertPlacement(DynInstDataOpInterface op, PhysLocationAttr);

  /// Load the database from the IR. Return the number of placements which
  /// failed to load due to invalid specifications.
  size_t addDesignPlacements();

  /// Remove the placement from the DB.
  void removePlacement(DynInstDataOpInterface, PhysLocationAttr);

  /// Check to make sure the move is going to succeed.
  LogicalResult movePlacementCheck(DynInstDataOpInterface op,
                                   PhysLocationAttr from, PhysLocationAttr to);
  /// Move it.
  void movePlacement(DynInstDataOpInterface op, PhysLocationAttr from,
                     PhysLocationAttr to);
};

} // namespace msft
} // namespace circt

#endif // CIRCT_DIALECT_MSFT_DEVICEDB_H
