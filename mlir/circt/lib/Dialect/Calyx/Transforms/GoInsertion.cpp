//===- GoInsertion.cpp - Go Insertion Pass ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the Go Insertion pass.
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

namespace {

struct GoInsertionPass : public GoInsertionBase<GoInsertionPass> {
  void runOnOperation() override;
};

} // end anonymous namespace

void GoInsertionPass::runOnOperation() {
  ComponentOp component = getOperation();
  auto wiresOp = component.getWiresOp();

  OpBuilder builder(wiresOp->getRegion(0));
  auto undefinedOp =
      builder.create<UndefinedOp>(wiresOp->getLoc(), builder.getI1Type());

  wiresOp.walk([&](GroupOp group) {
    OpBuilder builder(group->getRegion(0));
    // Since the source of a GroupOp's go signal isn't set until the
    // the Compile Control pass, use an undefined value.
    auto goOp = builder.create<GroupGoOp>(group->getLoc(), undefinedOp);

    updateGroupAssignmentGuards(builder, group, goOp);
  });
}

std::unique_ptr<mlir::Pass> circt::calyx::createGoInsertionPass() {
  return std::make_unique<GoInsertionPass>();
}
