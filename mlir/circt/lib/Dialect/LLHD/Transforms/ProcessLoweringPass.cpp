//===- ProcessLoweringPass.cpp - Implement Process Lowering Pass ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implement Pass to transform combinational processes to entities.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/LLHD/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Visitors.h"

using namespace mlir;
using namespace circt;

namespace {

struct ProcessLoweringPass
    : public llhd::ProcessLoweringBase<ProcessLoweringPass> {
  void runOnOperation() override;
};

/// Backtrack a signal value and make sure that every part of it is in the
/// observer list at some point. Assumes that there is no operation that adds
/// parts to a signal that it does not take as input (e.g. something like
/// llhd.sig.zext %sig : !llhd.sig<i32> -> !llhd.sig<i64>).
static LogicalResult checkSignalsAreObserved(OperandRange obs, Value value) {
  // If the value in the observer list, we don't need to backtrack further.
  if (llvm::is_contained(obs, value))
    return success();

  if (Operation *op = value.getDefiningOp()) {
    // If no input is a signal, this operation creates one and thus this is the
    // last point where it could have been observed. As we've already checked
    // that, we can fail here. This includes for example llhd.sig
    if (llvm::none_of(op->getOperands(), [](Value arg) {
          return arg.getType().isa<llhd::SigType>();
        }))
      return failure();

    // Only recusively backtrack signal values. Other values cannot be changed
    // from outside or with a delay. If they come from probes at some point,
    // they are covered by that probe. As soon as we find a signal that is not
    // observed no matter how far we backtrack, we fail.
    return success(llvm::all_of(op->getOperands(), [&](Value arg) {
      return !arg.getType().isa<llhd::SigType>() ||
             succeeded(checkSignalsAreObserved(obs, arg));
    }));
  }

  // If the value is a module argument (no block arguments except for the entry
  // block are allowed here) and was not observed, we cannot backtrack further
  // and thus fail.
  return failure();
}

static LogicalResult isProcValidToLower(llhd::ProcOp op) {
  size_t numBlocks = op.getBody().getBlocks().size();

  if (numBlocks == 1) {
    if (!isa<llhd::HaltOp>(op.getBody().back().getTerminator()))
      return op.emitOpError("during process-lowering: entry block is required "
                            "to be terminated by llhd.halt");
    return success();
  }

  if (numBlocks == 2) {
    Block &first = op.getBody().front();
    Block &last = op.getBody().back();

    if (last.getArguments().size() != 0)
      return op.emitOpError(
          "during process-lowering: the second block (containing the "
          "llhd.wait) is not allowed to have arguments");

    if (!isa<cf::BranchOp>(first.getTerminator()))
      return op.emitOpError("during process-lowering: the first block has to "
                            "be terminated by a cf.br operation");

    if (auto wait = dyn_cast<llhd::WaitOp>(last.getTerminator())) {
      // No optional time argument is allowed
      if (wait.getTime())
        return wait.emitOpError(
            "during process-lowering: llhd.wait terminators with optional time "
            "argument cannot be lowered to structural LLHD");

      // Every probed signal has to occur in the observed signals list in
      // the wait instruction
      WalkResult result = op.walk([&wait](llhd::PrbOp prbOp) -> WalkResult {
        if (failed(checkSignalsAreObserved(wait.getObs(), prbOp.getSignal())))
          return wait.emitOpError(
              "during process-lowering: the wait terminator is required to "
              "have all probed signals as arguments");

        return WalkResult::advance();
      });
      return failure(result.wasInterrupted());
    }

    return op.emitOpError("during process-lowering: the second block must be "
                          "terminated by llhd.wait");
  }

  return op.emitOpError(
      "process-lowering only supports processes with either one basic block "
      "terminated by a llhd.halt operation or two basic blocks where the first "
      "one contains a cf.br terminator and the second one is terminated by a "
      "llhd.wait operation");
}

void ProcessLoweringPass::runOnOperation() {
  ModuleOp module = getOperation();

  WalkResult result = module.walk([](llhd::ProcOp op) -> WalkResult {
    // Check invariants
    if (failed(isProcValidToLower(op)))
      return WalkResult::interrupt();

    OpBuilder builder(op);

    // Replace proc with entity
    llhd::EntityOp entity =
        builder.create<llhd::EntityOp>(op.getLoc(), op.getFunctionType(),
                                       op.getIns(), /*argAttrs=*/ArrayAttr(),
                                       /*resAttrs=*/ArrayAttr());
    // Set the symbol name of the entity to the same as the process (as the
    // process gets deleted anyways).
    entity.setName(op.getName());
    // Move all blocks from the process to the entity, the process does not have
    // a region afterwards.
    entity.getBody().takeBody(op.getBody());
    // In the case that wait is used to suspend the process, we need to merge
    // the two blocks as we needed the second block to have a target for wait
    // (the entry block cannot be targeted).
    if (entity.getBody().getBlocks().size() == 2) {
      Block &first = entity.getBody().front();
      Block &second = entity.getBody().back();
      // Delete the BranchOp operation in the entry block
      first.getTerminator()->dropAllReferences();
      first.getTerminator()->erase();
      // Move operations of second block in entry block.
      first.getOperations().splice(first.end(), second.getOperations());
      // Drop all references to the second block and delete it.
      second.dropAllReferences();
      second.dropAllDefinedValueUses();
      second.erase();
    }

    // Delete the process as it is now replaced by an entity.
    op->dropAllReferences();
    op->dropAllDefinedValueUses();
    op->erase();

    // Remove the remaining llhd.halt or llhd.wait terminator
    Operation *terminator = entity.getBody().front().getTerminator();
    terminator->dropAllReferences();
    terminator->dropAllUses();
    terminator->erase();

    return WalkResult::advance();
  });

  if (result.wasInterrupted())
    signalPassFailure();
}
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
circt::llhd::createProcessLoweringPass() {
  return std::make_unique<ProcessLoweringPass>();
}
