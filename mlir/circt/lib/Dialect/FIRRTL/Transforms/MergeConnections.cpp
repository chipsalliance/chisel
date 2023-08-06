//===- MergeConnections.cpp - Merge expanded connections --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This pass merges expanded connections into one connection.
// LowerTypes fully expands aggregate connections even when semantically
// not necessary to expand because it is required for ExpandWhen.
//
// More specifically this pass folds the following patterns:
//   %dest(0) <= v0
//   %dest(1) <= v1
//   ...
//   %dest(n) <= vn
// into
//   %dest <= {vn, .., v1, v0}
// Also if v0, v1, .., vn are subfield op like %a(0), %a(1), ..., a(n), then we
// merge entire connections into %dest <= %a.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "firrtl-merge-connections"

using namespace circt;
using namespace firrtl;

// Return true if value is essentially constant.
static bool isConstantLike(Value value) {
  if (isa_and_nonnull<ConstantOp, InvalidValueOp>(value.getDefiningOp()))
    return true;
  if (auto bitcast = value.getDefiningOp<BitCastOp>())
    return isConstant(bitcast.getInput());

  // TODO: Add unrealized_conversion, asUInt, asSInt
  return false;
}

namespace {
//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

// A helper struct to merge connections.
struct MergeConnection {
  MergeConnection(FModuleOp moduleOp, bool enableAggressiveMerging)
      : moduleOp(moduleOp), enableAggressiveMerging(enableAggressiveMerging) {}

  // Return true if something is changed.
  bool run();
  bool changed = false;

  // Return true if the given connect op is merged.
  bool peelConnect(StrictConnectOp connect);

  // A map from a destination FieldRef to a pair of (i) the number of
  // connections seen so far and (ii) the vector to store subconnections.
  DenseMap<FieldRef, std::pair<unsigned, SmallVector<StrictConnectOp>>>
      connections;

  FModuleOp moduleOp;
  ImplicitLocOpBuilder *builder = nullptr;

  // If true, we merge connections even when source values will not be
  // simplified.
  bool enableAggressiveMerging = false;
};

bool MergeConnection::peelConnect(StrictConnectOp connect) {
  // Ignore connections between different types because it will produce a
  // partial connect. Also ignore non-passive connections or non-integer
  // connections.
  LLVM_DEBUG(llvm::dbgs() << "Visiting " << connect << "\n");
  auto destTy = type_dyn_cast<FIRRTLBaseType>(connect.getDest().getType());
  if (!destTy || !destTy.isPassive() ||
      !firrtl::getBitWidth(destTy).has_value())
    return false;

  auto destFieldRef = getFieldRefFromValue(connect.getDest());
  auto destRoot = destFieldRef.getValue();

  // If dest is derived from mem op or has a ground type, we cannot merge them.
  // If the connect's destination is a root value, we cannot merge.
  if (destRoot.getDefiningOp<MemOp>() || destRoot == connect.getDest())
    return false;

  Value parent;
  unsigned index;
  if (auto subfield = dyn_cast<SubfieldOp>(connect.getDest().getDefiningOp()))
    parent = subfield.getInput(), index = subfield.getFieldIndex();
  else if (auto subindex =
               dyn_cast<SubindexOp>(connect.getDest().getDefiningOp()))
    parent = subindex.getInput(), index = subindex.getIndex();
  else
    llvm_unreachable("unexpected destination");

  auto &countAndSubConnections = connections[getFieldRefFromValue(parent)];
  auto &count = countAndSubConnections.first;
  auto &subConnections = countAndSubConnections.second;

  // If it is the first time to visit the parent op, then allocate the vector
  // for subconnections.
  if (count == 0) {
    if (auto bundle = type_dyn_cast<BundleType>(parent.getType()))
      subConnections.resize(bundle.getNumElements());
    if (auto vector = type_dyn_cast<FVectorType>(parent.getType()))
      subConnections.resize(vector.getNumElements());
  }
  ++count;
  subConnections[index] = connect;

  // If we haven't visited all subconnections, stop at this point.
  if (count != subConnections.size())
    return false;

  changed = true;

  auto parentType = parent.getType();

  auto getMergedValue = [&](auto aggregateType) {
    SmallVector<Value> operands;

    // This flag tracks whether we can use the parent of source values as the
    // merged value.
    bool canUseSourceParent = true;
    bool areOperandsAllConstants = true;

    // The value which might be used as a merged value.
    Value sourceParent;

    auto checkSourceParent = [&](auto subelement, unsigned destIndex,
                                 unsigned sourceIndex) {
      // In the first iteration, register a parent value.
      if (destIndex == 0) {
        if (subelement.getInput().getType() == parentType)
          sourceParent = subelement.getInput();
        else {
          // If types are not same, it is not possible to use it.
          canUseSourceParent = false;
        }
      }

      // Check that input is the same as `sourceAggregate` and indexes match.
      canUseSourceParent &=
          subelement.getInput() == sourceParent && destIndex == sourceIndex;
    };

    for (auto idx : llvm::seq(0u, (unsigned)aggregateType.getNumElements())) {
      auto src = subConnections[idx].getSrc();
      assert(src && "all subconnections are guranteed to exist");
      operands.push_back(src);

      areOperandsAllConstants &= isConstantLike(src);

      // From here, check whether the value is derived from the same aggregate
      // value.

      // If canUseSourceParent is already false, abort.
      if (!canUseSourceParent)
        continue;

      // If the value is an argument, it is not derived from an aggregate value.
      if (!src.getDefiningOp()) {
        canUseSourceParent = false;
        continue;
      }

      TypeSwitch<Operation *>(src.getDefiningOp())
          .template Case<SubfieldOp>([&](SubfieldOp subfield) {
            checkSourceParent(subfield, idx, subfield.getFieldIndex());
          })
          .template Case<SubindexOp>([&](SubindexOp subindex) {
            checkSourceParent(subindex, idx, subindex.getIndex());
          })
          .Default([&](auto) { canUseSourceParent = false; });
    }

    // If it is fine to use `sourceParent` as a merged value, we just
    // return it.
    if (canUseSourceParent) {
      LLVM_DEBUG(llvm::dbgs() << "Success to merge " << destFieldRef.getValue()
                              << " ,fieldID= " << destFieldRef.getFieldID()
                              << " to " << sourceParent << "\n";);
      // Erase connections except for subConnections[index] since it must be
      // erased at the top-level loop.
      for (auto idx : llvm::seq(0u, static_cast<unsigned>(operands.size())))
        if (idx != index)
          subConnections[idx].erase();
      return sourceParent;
    }

    // If operands are not all constants, we don't merge connections unless
    // "aggressive-merging" option is enabled.
    if (!enableAggressiveMerging && !areOperandsAllConstants)
      return Value();

    SmallVector<Location> locs;
    // Otherwise, we concat all values and cast them into the aggregate type.
    for (auto idx : llvm::seq(0u, static_cast<unsigned>(operands.size()))) {
      locs.push_back(subConnections[idx].getLoc());
      // Erase connections except for subConnections[index] since it must be
      // erased at the top-level loop.
      if (idx != index)
        subConnections[idx].erase();
    }

    return isa<FVectorType>(parentType)
               ? builder->createOrFold<VectorCreateOp>(
                     builder->getFusedLoc(locs), parentType, operands)
               : builder->createOrFold<BundleCreateOp>(
                     builder->getFusedLoc(locs), parentType, operands);
  };

  Value merged;
  if (auto bundle = dyn_cast_or_null<BundleType>(parentType))
    merged = getMergedValue(bundle);
  if (auto vector = dyn_cast_or_null<FVectorType>(parentType))
    merged = getMergedValue(vector);
  if (!merged)
    return false;

  // Emit strict connect if possible, fallback to normal connect.
  // Don't use emitConnect(), will split the connect apart.
  auto parentBaseTy = type_cast<FIRRTLBaseType>(parentType);
  if (parentBaseTy.isPassive() && !parentBaseTy.hasUninferredWidth())
    builder->create<StrictConnectOp>(connect.getLoc(), parent, merged);
  else
    builder->create<ConnectOp>(connect.getLoc(), parent, merged);

  return true;
}

bool MergeConnection::run() {
  ImplicitLocOpBuilder theBuilder(moduleOp.getLoc(), moduleOp.getContext());
  builder = &theBuilder;
  auto *body = moduleOp.getBodyBlock();
  // Merge connections by forward iterations.
  for (auto it = body->begin(), e = body->end(); it != e;) {
    auto connectOp = dyn_cast<StrictConnectOp>(*it);
    if (!connectOp) {
      it++;
      continue;
    }
    builder->setInsertionPointAfter(connectOp);
    builder->setLoc(connectOp.getLoc());
    bool removeOp = peelConnect(connectOp);
    ++it;
    if (removeOp)
      connectOp.erase();
  }

  // Clean up dead operations introduced by this pass.
  for (auto &op : llvm::make_early_inc_range(llvm::reverse(*body)))
    if (isa<SubfieldOp, SubindexOp, InvalidValueOp, ConstantOp, BitCastOp,
            CatPrimOp>(op))
      if (op.use_empty()) {
        changed = true;
        op.erase();
      }

  return changed;
}

struct MergeConnectionsPass
    : public MergeConnectionsBase<MergeConnectionsPass> {
  MergeConnectionsPass(bool enableAggressiveMergingFlag) {
    enableAggressiveMerging = enableAggressiveMergingFlag;
  }
  void runOnOperation() override;
};

} // namespace

void MergeConnectionsPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "===----- Running MergeConnections "
                             "--------------------------------------===\n"
                          << "Module: '" << getOperation().getName() << "'\n";);

  MergeConnection mergeConnection(getOperation(), enableAggressiveMerging);
  bool changed = mergeConnection.run();

  if (!changed)
    return markAllAnalysesPreserved();
}

std::unique_ptr<mlir::Pass>
circt::firrtl::createMergeConnectionsPass(bool enableAggressiveMerging) {
  return std::make_unique<MergeConnectionsPass>(enableAggressiveMerging);
}
