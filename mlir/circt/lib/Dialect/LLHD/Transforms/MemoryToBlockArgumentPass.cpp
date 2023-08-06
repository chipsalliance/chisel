//===- VarToBlockArgumentPass.cpp - Implement Var to Block Argument Pass --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implement pass to promote memory to block arguments.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/LLHD/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include <set>

using namespace circt;

namespace {

struct MemoryToBlockArgumentPass
    : public llhd::MemoryToBlockArgumentBase<MemoryToBlockArgumentPass> {
  void runOnOperation() override;
};

} // anonymous namespace

/// Add the dominance fontier blocks of 'frontierOf' to the 'df' set
static void getDominanceFrontier(Block *frontierOf, Operation *op,
                                 std::set<Block *> &df) {
  mlir::DominanceInfo dom(op);
  for (Block &block : op->getRegion(0).getBlocks()) {
    for (Block *pred : block.getPredecessors()) {
      if (dom.dominates(frontierOf, pred) &&
          !dom.properlyDominates(frontierOf, &block) &&
          block.getOps<llhd::VarOp>().empty()) {
        df.insert(&block);
      }
    }
  }
}

/// Add the blocks in the closure of the dominance fontier relation of all the
/// block in 'initialSet' to 'closure'
static void getDFClosure(SmallVectorImpl<Block *> &initialSet, Operation *op,
                         std::set<Block *> &closure) {
  unsigned numElements;
  for (Block *block : initialSet) {
    getDominanceFrontier(block, op, closure);
  }
  do {
    numElements = closure.size();
    for (Block *block : closure) {
      getDominanceFrontier(block, op, closure);
    }
  } while (numElements < closure.size());
}

/// Add a block argument to a given terminator. Only 'std.br', 'std.cond_br' and
/// 'llhd.wait' are supported. The successor block has to be provided for the
/// 'std.cond_br' terminator which has two possible successors.
static void addBlockOperandToTerminator(Operation *terminator,
                                        Block *successsor, Value toAppend) {
  if (auto wait = dyn_cast<llhd::WaitOp>(terminator)) {
    wait.getDestOpsMutable().append(toAppend);
  } else if (auto br = dyn_cast<mlir::cf::BranchOp>(terminator)) {
    br.getDestOperandsMutable().append(toAppend);
  } else if (auto condBr = dyn_cast<mlir::cf::CondBranchOp>(terminator)) {
    if (condBr.getFalseDest() == successsor) {
      condBr.getFalseDestOperandsMutable().append(toAppend);
    } else {
      condBr.getTrueDestOperandsMutable().append(toAppend);
    }
  } else {
    llvm_unreachable("unsupported terminator op");
  }
}

void MemoryToBlockArgumentPass::runOnOperation() {
  Operation *operation = getOperation();
  OpBuilder builder(operation);

  // No operations that have their own region and are not isolated from above
  // are allowed for now.
  WalkResult result = operation->walk([](Operation *op) -> WalkResult {
    if (op->getNumRegions() > 0 &&
        !op->hasTrait<OpTrait::IsIsolatedFromAbove>())
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return;

  // Get all variables defined in the body of this operation
  // Note that variables that are passed as a function argument are not
  // considered.
  SmallVector<Value, 16> vars;
  for (llhd::VarOp var : operation->getRegion(0).getOps<llhd::VarOp>()) {
    vars.push_back(var.getResult());
  }

  // Don't consider variables that are used in other operations than load and
  // store (e.g. as an argument to a call)
  for (auto var = vars.begin(); var != vars.end(); ++var) {
    for (Operation *user : var->getUsers()) {
      if (!isa<llhd::LoadOp>(user) && !isa<llhd::StoreOp>(user)) {
        vars.erase(var--);
        break;
      }
    }
  }

  // For each variable find the blocks where a value is stored to it
  for (Value var : vars) {
    SmallVector<Block *, 16> defBlocks;
    defBlocks.push_back(
        var.getDefiningOp<llhd::VarOp>().getOperation()->getBlock());
    operation->walk([&](llhd::StoreOp op) {
      if (op.getPointer() == var)
        defBlocks.push_back(op.getOperation()->getBlock());
    });
    // Remove duplicates from the list
    std::sort(defBlocks.begin(), defBlocks.end());
    defBlocks.erase(std::unique(defBlocks.begin(), defBlocks.end()),
                    defBlocks.end());

    // Calculate initial set of join points
    std::set<Block *> joinPoints;
    getDFClosure(defBlocks, operation, joinPoints);

    for (Block *jp : joinPoints) {
      // Add a block argument for the variable at each join point
      BlockArgument phi = jp->addArgument(
          var.getType().cast<llhd::PtrType>().getUnderlyingType(),
          var.getLoc());

      // Add a load at the end of every predecessor and pass the loaded value as
      // the block argument
      for (Block *pred : jp->getPredecessors()) {
        // Set insertion point before terminator to insert the load operation
        builder.setInsertionPoint(pred->getTerminator());
        Value load = builder.create<llhd::LoadOp>(
            pred->getTerminator()->getLoc(),
            var.getType().cast<llhd::PtrType>().getUnderlyingType(), var);
        // Add the loaded value as additional block argument
        addBlockOperandToTerminator(pred->getTerminator(), jp, load);
      }
      // Insert a store at the beginning of the join point to make removal of
      // all the memory operations easier later on
      builder.setInsertionPointToStart(jp);
      builder.create<llhd::StoreOp>(jp->front().getLoc(), var, phi);
    }

    // Basically reaching definitions analysis and replacing the loaded values
    // by the values stored
    DenseMap<Block *, Value> outputMap;
    SmallPtrSet<Block *, 32> workQueue;
    SmallPtrSet<Block *, 32> workDone;

    workQueue.insert(&operation->getRegion(0).front());

    while (!workQueue.empty()) {
      // Pop block to process at the front
      Block *block = *workQueue.begin();
      workQueue.erase(block);

      // Remember the currently stored value
      Value currStoredValue;

      // Update the value stored for the current variable at the start of this
      // block
      for (Block *pred : block->getPredecessors()) {
        // Get the value at the end of a predecessor block and set it to the
        // currently stored value at the start of this block
        // NOTE: because we added block arguments and a store at the beginning
        // of each join point we can assume that all predecessors have the same
        // value stored at their end here because if that is not the case the
        // first instruction in this block is a store instruction and will
        // update the currently stored value to a correct one
        if (!currStoredValue && outputMap.count(pred)) {
          currStoredValue = outputMap[pred];
          break;
        }
      }

      // Iterate through all operations of the current block
      for (auto op = block->begin(); op != block->end(); ++op) {
        // Update currStoredValue at every store operation that stores to the
        // variable we are currently considering
        if (auto store = dyn_cast<llhd::StoreOp>(op)) {
          if (store.getPointer() == var) {
            currStoredValue = store.getValue();
            op = std::prev(op);
            store.getOperation()->dropAllReferences();
            store.erase();
          }
          // Set currStoredValue to the initializer value of the variable
          // operation that created the variable we are currently considering,
          // note that before that currStoredValue is uninitialized
        } else if (auto varOp = dyn_cast<llhd::VarOp>(op)) {
          if (varOp.getResult() == var)
            currStoredValue = varOp.getInit();
          // Replace the value returned by a load from the variable we are
          // currently considering with the currStoredValue and delete the load
          // operation
        } else if (auto load = dyn_cast<llhd::LoadOp>(op)) {
          if (load.getPointer() == var && currStoredValue) {
            op = std::prev(op);
            load.getResult().replaceAllUsesWith(currStoredValue);
            load.getOperation()->dropAllReferences();
            load.erase();
          }
        }
      }

      if (currStoredValue)
        outputMap.insert(std::make_pair(block, currStoredValue));

      workDone.insert(block);

      // Add all successors of this block to the work queue if they are not
      // already processed
      for (Block *succ : block->getSuccessors()) {
        if (!workDone.count(succ))
          workQueue.insert(succ);
      }
    }
  }

  // Remove all variable declarations for which we already removed all loads and
  // stores
  for (Value var : vars) {
    Operation *op = var.getDefiningOp();
    op->dropAllDefinedValueUses();
    op->dropAllReferences();
    op->erase();
  }
}

std::unique_ptr<OperationPass<llhd::ProcOp>>
circt::llhd::createMemoryToBlockArgumentPass() {
  return std::make_unique<MemoryToBlockArgumentPass>();
}
