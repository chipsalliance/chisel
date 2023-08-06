//===- MuxToControlFlow.cpp - Implement the MuxToControlFlow Pass ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implement a pass to convert muxes to control flow branches whenever it is
// beneficial for performance (i.e., when expected work avoided is more than
// branching costs)
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/Arc/ArcInterfaces.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arc-mux-to-control-flow"

using namespace circt;
using namespace arc;

//===----------------------------------------------------------------------===//
// MuxToControlFlow pass declarations
//===----------------------------------------------------------------------===//

namespace {

/// Convert muxes to if-statements.
struct MuxToControlFlowPass
    : public MuxToControlFlowBase<MuxToControlFlowPass> {
  MuxToControlFlowPass() = default;
  MuxToControlFlowPass(const MuxToControlFlowPass &pass)
      : MuxToControlFlowPass() {}

  void runOnOperation() override;

  Statistic numMuxesConverted{
      this, "num-muxes-converted",
      "Number of muxes that were converted to if-statements"};
  Statistic numMuxesRetained{this, "num-muxes-retained",
                             "Number of muxes that were not converted"};
};

/// Abstract over muxes to easy addition of support for other operations.
struct BranchInfo {
  BranchInfo() = default;
  BranchInfo(Value condition, Value trueValue, Value falseValue)
      : condition(condition), trueValue(trueValue), falseValue(falseValue) {}

  Value condition;
  Value trueValue;
  Value falseValue;

  operator bool() { return condition && trueValue && falseValue; }
};

} // namespace

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Check whether @param curr is valid to be moved into the if-branch, which is
/// the stopping condition of the BFS traversal.
static bool isValidToProceedTraversal(Operation *mux, Operation *curr,
                                      Value useValue,
                                      SmallPtrSetImpl<Operation *> &visited) {
  for (auto res : curr->getResults()) {
    for (auto *user : res.getUsers()) {
      // The use-sites of all results have to be within the same branch, thus
      // already have to be visited already. The only exception is the first
      // operation in the branch used by the mux itself.
      if (!visited.contains(user) && user != mux)
        return false;

      // The second part of the special case mentioned above (because otherwise
      // we would also include the first operation of the other branch).
      if (user == mux && res != useValue)
        return false;

      if (user->getBlock() != curr->getBlock())
        return false;
    }
  }

  return true;
}

/// Compute the set of operations that would only be used in the branch
/// represented by @param useValue.
static void computeFanIn(Operation *mux, Value useValue,
                         SmallPtrSetImpl<Operation *> &visited) {
  auto *op = useValue.getDefiningOp();
  if (!op)
    return;

  SmallVector<Operation *> worklist{op};

  while (!worklist.empty()) {
    auto *curr = worklist.front();
    worklist.erase(worklist.begin());

    if (visited.contains(curr))
      continue;

    if (!isValidToProceedTraversal(mux, curr, useValue, visited))
      continue;

    visited.insert(curr);

    for (auto val : curr->getOperands()) {
      if (auto *defOp = val.getDefiningOp())
        worklist.push_back(defOp);
    }
  }
}

/// Clone ops that are used in both branches of an if-statement but not outside
/// of it. This is just here because of experimentation reasons. Doing this
/// might allow for better instruction scheduling to slightly reduce ISA
/// register pressure (however, it is currently too naive to only take the
/// beneficial situations), but it will increase binary size which is especially
/// bad when the hot part would otherwise fit in instruction cache (but doesn't
/// really matter when it doesn't fit anyways as there is no temporal locality
/// anyways).
[[maybe_unused]] static void
cloneOpsIntoBranchesWhenUsedInBoth(mlir::scf::IfOp ifOp) {
  // Iterate over all operations at the same nesting level as the if-statement
  // (not the operations inside the if-statement).
  for (auto &op : llvm::reverse(*ifOp->getBlock())) {
    if (op.getNumResults() == 0)
      continue;

    // Collect all users of the current operations results.
    SmallVector<Operation *> users;
    for (auto result : op.getResults())
      users.append(llvm::to_vector(result.getUsers()));

    auto parentsOfUsers =
        llvm::map_range(users, [](auto user) { return user->getParentOp(); });

    auto allUsersNestedInIf = llvm::any_of(parentsOfUsers, [&](auto *parent) {
      return !(isa<mlir::scf::IfOp>(parent) &&
               parent->getBlock() == op.getBlock());
    });

    // Check that all users of the results are nested inside the same scf.if
    // operation
    if (allUsersNestedInIf || !llvm::all_equal(parentsOfUsers))
      continue;

    DenseMap<Region *, Value> cloneMap;
    for (auto &use : llvm::make_early_inc_range(op.getUses())) {
      auto *parentRegion = use.getOwner()->getParentRegion();
      if (!cloneMap.count(parentRegion)) {
        OpBuilder builder(&parentRegion->front().front());
        cloneMap[parentRegion] = builder.clone(op)->getResult(0);
      }
      use.set(cloneMap[parentRegion]);
    }
  }
}

/// Perform the actual conversion. Create the if-statement, move the operations
/// in its regions and delete the mux.
static void doConversion(Operation *op, BranchInfo info,
                         const SmallPtrSetImpl<Operation *> &thenOps,
                         const SmallPtrSetImpl<Operation *> &elseOps) {
  if (op->getNumResults() != 1)
    return;

  // Build the scf.if operation with the scf.yields inside.
  ImplicitLocOpBuilder builder(op->getLoc(), op);
  mlir::scf::IfOp ifOp = builder.create<mlir::scf::IfOp>(
      info.condition,
      [&](OpBuilder &builder, Location loc) {
        builder.create<mlir::scf::YieldOp>(loc, info.trueValue);
      },
      [&](OpBuilder &builder, Location loc) {
        builder.create<mlir::scf::YieldOp>(loc, info.falseValue);
      });

  op->getResult(0).replaceAllUsesWith(ifOp.getResult(0));

  for (auto &ops :
       llvm::make_early_inc_range(op->getParentRegion()->getOps())) {
    // Move operations into the then-branch if they are only used in there.
    // The original lexicographical order is preserved.
    if (thenOps.contains(&ops))
      ops.moveBefore(ifOp.thenBlock()->getTerminator());

    // Move operations into the else-branch if they are only used in there.
    // The original lexicographical order is preserved.
    if (elseOps.contains(&ops))
      ops.moveBefore(ifOp.elseBlock()->getTerminator());
  }

  op->erase();

  // NOTE: this is just here for some experimentation purposes
  // cloneOpsIntoBranchesWhenUsedInBoth(ifOp);
}

/// Simple helper to invoke the runtime cost interface for every operation in a
/// set and sum up the costs.
static uint32_t getCostEstimate(const SmallPtrSetImpl<Operation *> &ops) {
  uint32_t cost = 0;

  for (auto *op : ops) {
    if (auto *runtimeCostIF =
            dyn_cast<RuntimeCostEstimateDialectInterface>(op->getDialect())) {
      cost += runtimeCostIF->getCostEstimate(op);
    } else {
      LLVM_DEBUG(llvm::dbgs() << "No runtime cost estimate was provided for '"
                              << op->getName() << "', using default of 10\n");
      cost += 10;
    }
  }

  return cost;
}

//===----------------------------------------------------------------------===//
// Decision functions (configure the pass here)
//===----------------------------------------------------------------------===//

/// Convert concrete operations that should be converted to if-statements to a
/// more abstract representation the rest of the pass works with. This is the
/// place where support for more operations can be added (nothing else has to be
/// changed).
static BranchInfo getConversionInfo(Operation *op) {
  if (auto mux = dyn_cast<comb::MuxOp>(op))
    return BranchInfo{mux.getCond(), mux.getTrueValue(), mux.getFalseValue()};

  // TODO: we can also check for arith.select or other operations here

  return {};
}

/// Use the cost measure of each branch to heuristically decide whether to
/// actually perform the conversion.
/// TODO: improve and fine-tune this
static bool isBeneficialToConvert(Operation *op,
                                  const SmallPtrSetImpl<Operation *> &thenOps,
                                  const SmallPtrSetImpl<Operation *> &elseOps) {
  const uint32_t thenCost = getCostEstimate(thenOps);
  const uint32_t elseCost = getCostEstimate(elseOps);

  // Due to the nature of mux sequences we need to make sure that a reasonable
  // amount of operations stay in each if-branch because otherwise we end up
  // with if-statements that only contain anther if-statement, which is usually
  // more costly than keeping some muxes unconverted.
  if (auto parent = op->getParentOfType<mlir::scf::IfOp>()) {
    SmallPtrSet<Operation *, 32> ifBranchOps;

    for (auto &nestedOp : *op->getBlock()) {
      if (!thenOps.contains(&nestedOp) && !elseOps.contains(&nestedOp))
        ifBranchOps.insert(&nestedOp);
    }

    if (getCostEstimate(ifBranchOps) < 100)
      return false;
  }

  // return thenCost + elseCost >= 100 && (thenCost == 0 || elseCost == 0);
  return (thenCost >= 100 || thenCost == 0) &&
         (elseCost >= 100 || elseCost == 0) &&
         std::abs((int)thenCost - (int)elseCost) >= 100;
}

//===----------------------------------------------------------------------===//
// MuxToControlFlow pass definitions
//===----------------------------------------------------------------------===//

// FIXME: Assumes that the regions in which muxes exist are topologically
// ordered.
// FIXME: does not consider side-effects
void MuxToControlFlowPass::runOnOperation() {
  // Collect all operations that support the conversion to scf.if operations.
  // Use 'walk' instead of 'getOps' as we also want to visit nested regions.
  // We need to collect them because moving ops while iterating over them
  // would require complicated iterator advancing/skipping but also tracking
  // back to not miss supported operations.
  SmallVector<Operation *> supportedOps;
  getOperation()->walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    // Skip ops with graph regions and ops that can contain ops with write
    // semantics for now until side-effects and topological ordering is properly
    // handled.
    if (isa<hw::HWModuleOp, arc::ModelOp>(op))
      return WalkResult::skip();

    if (getConversionInfo(op))
      supportedOps.push_back(op);

    return WalkResult::advance();
  });

  // We want to visit the operations bottom-up to visit the operations with the
  // longest fan-in first. However, the other direction would also work with the
  // current implementation.
  for (auto *op : llvm::reverse(supportedOps)) {
    auto info = getConversionInfo(op);

    // Compute the operations in the fan-in of each branch and use them to
    // decide whether the operation should be converted.
    // Stop at the first value that's also used outside of the branch.
    llvm::SmallPtrSet<Operation *, 32> thenOps, elseOps;
    computeFanIn(op, info.trueValue, thenOps);
    computeFanIn(op, info.falseValue, elseOps);

    // Apply a cost measure to the operations in the branches and only convert
    // when a performance increase can be expected.
    if (isBeneficialToConvert(op, thenOps, elseOps)) {
      doConversion(op, info, thenOps, elseOps);
      ++numMuxesConverted;
    } else {
      ++numMuxesRetained;
    }
  }
}

std::unique_ptr<Pass> arc::createMuxToControlFlowPass() {
  return std::make_unique<MuxToControlFlowPass>();
}
