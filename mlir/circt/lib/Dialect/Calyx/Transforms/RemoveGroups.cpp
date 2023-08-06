//===- RemoveGroups.cpp - Remove Groups Pass --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the Remove Groups pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/Calyx/CalyxHelpers.h"
#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Dialect/Calyx/CalyxPasses.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"

using namespace circt;
using namespace calyx;
using namespace mlir;

/// Makes several modifications to the operations of a GroupOp:
/// 1. Assign the 'done' signal of the component with the done_op of the top
///    level control group.
/// 2. Append the 'go' signal of the component to guard of each assignment.
/// 3. Replace all uses of GroupGoOp with the respective guard, and delete the
///    GroupGoOp.
/// 4. Remove the GroupDoneOp.
static void modifyGroupOperations(ComponentOp component) {
  auto control = component.getControlOp();
  // Get the only EnableOp in the control.
  auto topLevel = *control.getRegion().getOps<EnableOp>().begin();
  auto topLevelName = topLevel.getGroupName();

  auto wires = component.getWiresOp();
  Value componentGoPort = component.getGoPort();
  wires.walk([&](GroupOp group) {
    auto &groupRegion = group->getRegion(0);
    OpBuilder builder(groupRegion);
    // Walk the assignments and append component's 'go' signal to each guard.
    updateGroupAssignmentGuards(builder, group, componentGoPort);

    auto groupDone = group.getDoneOp();
    if (topLevelName == group.getSymName()) {
      // Replace `calyx.group_done %0, %1 ? : i1`
      //    with `calyx.assign %done, %0, %1 ? : i1`
      auto assignOp =
          builder.create<AssignOp>(group->getLoc(), component.getDonePort(),
                                   groupDone.getSrc(), groupDone.getGuard());
      groupDone->replaceAllUsesWith(assignOp);
    } else {
      // Replace calyx.group_go's uses with its guard, e.g.
      //    %A.go = calyx.group_go %true, %3 ? : i1
      //    %0 = comb.and %1, %A.go : i1
      //    ->
      //    %0 = comb.and %1, %3 : i1
      auto groupGo = group.getGoOp();
      auto groupGoGuard = groupGo.getGuard();
      groupGo.replaceAllUsesWith(groupGoGuard);
      groupGo->erase();
    }
    // In either case, remove the group's done value.
    groupDone->erase();
  });
}

/// Inlines each group in the WiresOp.
void inlineGroups(ComponentOp component) {
  auto &wiresRegion = component.getWiresOp().getRegion();
  auto &wireBlocks = wiresRegion.getBlocks();
  auto lastBlock = wiresRegion.end();

  // Inline the body of each group as a Block into the WiresOp.
  wiresRegion.walk([&](GroupOp group) {
    wireBlocks.splice(lastBlock, group.getRegion().getBlocks());
    group->erase();
  });

  // Merge the operations of each Block into the first block of the WiresOp.
  auto firstBlock = wireBlocks.begin();
  for (auto it = firstBlock, e = lastBlock; it != e; ++it) {
    if (it == firstBlock)
      continue;
    firstBlock->getOperations().splice(firstBlock->end(), it->getOperations());
  }

  // Erase the (now) empty blocks.
  while (&wiresRegion.front() != &wiresRegion.back())
    wiresRegion.back().erase();
}

namespace {

struct RemoveGroupsPass : public RemoveGroupsBase<RemoveGroupsPass> {
  void runOnOperation() override;
};

} // end anonymous namespace

void RemoveGroupsPass::runOnOperation() {
  ComponentOp component = getOperation();

  // Early exit if there is no control to compile.
  if (component.getControlOp().getOps().empty())
    return;

  // Make the necessary modifications to each group's operations.
  modifyGroupOperations(component);

  // Inline the body of each group.
  inlineGroups(component);

  // Remove the last EnableOp from the control.
  auto control = component.getControlOp();
  control.walk([&](EnableOp enable) { enable->erase(); });
}

std::unique_ptr<mlir::Pass> circt::calyx::createRemoveGroupsPass() {
  return std::make_unique<RemoveGroupsPass>();
}
