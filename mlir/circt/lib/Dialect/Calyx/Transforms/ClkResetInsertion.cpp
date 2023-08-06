//===- ClkResetInsertion.cpp - Clock & reset insertion pass -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Adds assignments from a components 'clk' and 'reset' port to every
// component that contains an input 'clk' or 'reset' port.
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

/// Iterates over the cells in 'comp' and connects any input port with attribute
/// 'portID' to 'fromPort'.
static void doPortPassthrough(ComponentOp comp, Value fromPort,
                              StringRef portID) {
  MLIRContext *ctx = comp.getContext();
  OpBuilder builder(ctx);
  builder.setInsertionPointToStart(comp.getWiresOp().getBodyBlock());

  for (auto cell : comp.getOps<CellInterface>()) {
    for (auto port : cell.getInputPorts()) {
      if (!cell.portInfo(port).hasAttribute(portID))
        continue;
      builder.create<AssignOp>(cell.getLoc(), port, fromPort);
    }
  }
}

struct ClkInsertionPass : public ClkInsertionBase<ClkInsertionPass> {
  void runOnOperation() override {
    doPortPassthrough(getOperation(), getOperation().getClkPort(), "clk");
  }
};

struct ResetInsertionPass : public ResetInsertionBase<ResetInsertionPass> {
  void runOnOperation() override {
    doPortPassthrough(getOperation(), getOperation().getResetPort(), "reset");
  }
};

} // end anonymous namespace

std::unique_ptr<mlir::Pass> circt::calyx::createClkInsertionPass() {
  return std::make_unique<ClkInsertionPass>();
}

std::unique_ptr<mlir::Pass> circt::calyx::createResetInsertionPass() {
  return std::make_unique<ResetInsertionPass>();
}
