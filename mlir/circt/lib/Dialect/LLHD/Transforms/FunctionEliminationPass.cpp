//===- FunctionEliminationPass.cpp - Implement Function Elimination Pass --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implement pass to check that all functions got inlined and delete them.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/LLHD/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Visitors.h"

using namespace circt;

namespace {

struct FunctionEliminationPass
    : public llhd::FunctionEliminationBase<FunctionEliminationPass> {
  void runOnOperation() override;
};

void FunctionEliminationPass::runOnOperation() {
  ModuleOp module = getOperation();

  WalkResult result = module.walk([](mlir::func::CallOp op) -> WalkResult {
    if (isa<llhd::ProcOp>(op->getParentOp()) ||
        isa<llhd::EntityOp>(op->getParentOp())) {
      return emitError(
          op.getLoc(),
          "Not all functions are inlined, there is at least "
          "one function call left within a llhd.proc or llhd.entity.");
    }
    return WalkResult::advance();
  });

  if (result.wasInterrupted()) {
    signalPassFailure();
    return;
  }

  module.walk([](mlir::func::FuncOp op) { op.erase(); });
}
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
circt::llhd::createFunctionEliminationPass() {
  return std::make_unique<FunctionEliminationPass>();
}
