//===- DCMaterialization.cpp - Fork/sink materialization pass ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the Fork/sink materialization pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/DC/DCOps.h"
#include "circt/Dialect/DC/DCPasses.h"

using namespace circt;
using namespace dc;
using namespace mlir;

static void replaceFirstUse(Operation *op, Value oldVal, Value newVal) {
  for (int i = 0, e = op->getNumOperands(); i < e; ++i)
    if (op->getOperand(i) == oldVal) {
      op->setOperand(i, newVal);
      break;
    }
}

static void insertSink(Value val, OpBuilder &rewriter) {
  rewriter.setInsertionPointAfterValue(val);
  rewriter.create<SinkOp>(val.getLoc(), val);
}

static void insertFork(Value result, OpBuilder &rewriter) {
  // Get successor operations
  std::vector<Operation *> opsToProcess;
  for (auto &u : result.getUses())
    opsToProcess.push_back(u.getOwner());

  // Insert fork after op
  rewriter.setInsertionPointAfterValue(result);
  auto forkSize = opsToProcess.size();
  auto newFork = rewriter.create<ForkOp>(result.getLoc(), result, forkSize);

  // Modify operands of successor
  // opsToProcess may have multiple instances of same operand
  // Replace uses one by one to assign different fork outputs to them
  for (auto [op, forkRes] : llvm::zip(opsToProcess, newFork->getResults()))
    replaceFirstUse(op, result, forkRes);
}

// Insert Fork Operation for every operation and function argument with more
// than one successor.
static LogicalResult addForkOps(Block &block, OpBuilder &rewriter) {
  for (Operation &op : block) {
    // Ignore terminators.
    if (!op.hasTrait<OpTrait::IsTerminator>()) {
      for (auto result : op.getResults()) {
        // If there is a result, it is used more than once, and it is a DC
        // type, fork it!
        if (!result.use_empty() && !result.hasOneUse() &&
            result.getType().isa<dc::TokenType, dc::ValueType>())
          insertFork(result, rewriter);
      }
    }
  }

  for (auto barg : block.getArguments())
    if (!barg.use_empty() && !barg.hasOneUse())
      insertFork(barg, rewriter);

  return success();
}

namespace circt {
namespace dc {

// Create sink for every unused result
LogicalResult addSinkOps(Block &block, OpBuilder &rewriter) {
  for (auto arg : block.getArguments()) {
    if (arg.use_empty())
      insertSink(arg, rewriter);
  }
  for (Operation &op : block) {
    if (op.getNumResults() == 0)
      continue;

    for (auto result : op.getResults())
      if (result.use_empty())
        insertSink(result, rewriter);
  }

  return success();
}

} // namespace dc
} // namespace circt

namespace {
struct DCMaterializeForksSinksPass
    : public DCMaterializeForksSinksBase<DCMaterializeForksSinksPass> {
  void runOnOperation() override {
    auto funcOp = getOperation();
    if (funcOp.isExternal())
      return;
    OpBuilder builder(funcOp);

    auto walkRes = funcOp.walk([&](mlir::Block *block) {
      if (addForkOps(*block, builder).failed() ||
          addSinkOps(*block, builder).failed())
        return WalkResult::interrupt();

      return WalkResult::advance();
    });

    if (walkRes.wasInterrupted())
      return signalPassFailure();
  };
};

struct DCDematerializeForksSinksPass
    : public DCDematerializeForksSinksBase<DCDematerializeForksSinksPass> {
  void runOnOperation() override {
    auto funcOp = getOperation();

    if (funcOp.isExternal())
      return;
    funcOp.walk([&](dc::SinkOp sinkOp) { sinkOp.erase(); });
    funcOp.walk([&](dc::ForkOp forkOp) {
      for (auto res : forkOp->getResults())
        res.replaceAllUsesWith(forkOp.getOperand());
      forkOp.erase();
    });
  };
};

} // namespace

std::unique_ptr<mlir::Pass> circt::dc::createDCMaterializeForksSinksPass() {
  return std::make_unique<DCMaterializeForksSinksPass>();
}

std::unique_ptr<mlir::Pass> circt::dc::createDCDematerializeForksSinksPass() {
  return std::make_unique<DCDematerializeForksSinksPass>();
}
