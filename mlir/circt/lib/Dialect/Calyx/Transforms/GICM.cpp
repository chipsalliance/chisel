//===- GICM.cpp - Group-invariant code motion pass --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass performs GICM (group-invariant code motion) of operations which are
// deemed to be invariant of the group in which they are placed.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Dialect/Calyx/CalyxPasses.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace circt;
using namespace calyx;
using namespace mlir;

namespace {

struct GroupInvariantCodeMotionPass
    : public GroupInvariantCodeMotionBase<GroupInvariantCodeMotionPass> {
  void runOnOperation() override {
    auto wires = getOperation().getWiresOp();
    for (auto groupOp : wires.getOps<GroupOp>()) {
      for (auto &op : llvm::make_early_inc_range(groupOp.getOps())) {
        if (isa<GroupDoneOp, AssignOp, GroupGoOp>(op))
          continue;
        op.moveBefore(wires.getBodyBlock(), wires.getBodyBlock()->begin());
      }
    }
  }
};

} // end anonymous namespace

std::unique_ptr<mlir::Pass> circt::calyx::createGroupInvariantCodeMotionPass() {
  return std::make_unique<GroupInvariantCodeMotionPass>();
}
