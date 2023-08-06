//===- EarlyCodeMotionPass.cpp - Implement Early Code Motion Pass ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implement pass to move allowed instructions as far up in the CFG as possible.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "TemporalRegions.h"
#include "circt/Dialect/LLHD/Transforms/Passes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Dominance.h"

using namespace circt;

namespace {
struct EarlyCodeMotionPass
    : public llhd::EarlyCodeMotionBase<EarlyCodeMotionPass> {
  void runOnOperation() override;
};
} // namespace

/// Calculate intersection of two vectors, returns a new vector
static SmallVector<Block *, 8> intersection(SmallVectorImpl<Block *> &v1,
                                            SmallVectorImpl<Block *> &v2) {
  SmallVector<Block *, 8> res;
  std::sort(v1.begin(), v1.end());
  std::sort(v2.begin(), v2.end());

  std::set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(),
                        std::back_inserter(res));
  return res;
}

void EarlyCodeMotionPass::runOnOperation() {
  llhd::ProcOp proc = getOperation();
  llhd::TemporalRegionAnalysis trAnalysis = llhd::TemporalRegionAnalysis(proc);
  mlir::DominanceInfo dom(proc);

  DenseMap<Block *, unsigned> entryDistance;
  SmallPtrSet<Block *, 32> workDone;
  SmallPtrSet<Block *, 32> workPending;

  Block &entryBlock = proc.getBody().front();
  workPending.insert(&entryBlock);
  entryDistance.insert(std::make_pair(&entryBlock, 0));

  while (!workPending.empty()) {
    Block *block = *workPending.begin();
    workPending.erase(block);
    workDone.insert(block);

    for (auto iter = block->getOperations().begin();
         iter != block->getOperations().end(); ++iter) {
      Operation &op = *iter;
      if (!isa<llhd::PrbOp>(op) && !isa<llhd::SigOp>(op) &&
          (!mlir::isMemoryEffectFree(&op) ||
           op.hasTrait<OpTrait::IsTerminator>()))
        continue;

      SmallVector<Block *, 8> validPlacements;
      // Initialize validPlacements to all blocks in the process
      for (Block &b : proc.getBlocks())
        validPlacements.push_back(&b);

      // Delete all blocks in validPlacements that are not common to all
      // operands
      for (Value operand : op.getOperands()) {
        SmallVector<Block *, 8> dominationSet;
        Block *instBlock = nullptr;
        if (BlockArgument arg = operand.dyn_cast<BlockArgument>()) {
          instBlock = arg.getParentBlock();
        } else {
          instBlock = operand.getDefiningOp()->getBlock();
        }

        for (Block &b : proc.getBlocks()) {
          if (dom.dominates(instBlock, &b))
            dominationSet.push_back(&b);
        }
        validPlacements = intersection(dominationSet, validPlacements);
      }

      // The probe instruction has to stay in the same temporal region
      if (isa<llhd::PrbOp>(op)) {
        SmallVector<Block *, 8> blocksInTR =
            trAnalysis.getBlocksInTR(trAnalysis.getBlockTR(block));
        validPlacements = intersection(validPlacements, blocksInTR);
      }

      if (validPlacements.empty())
        continue;

      // Move the instruction to the block which is the closest to the entry
      // block (and valid)
      unsigned minBBdist = -1;
      Block *minBB = nullptr;
      for (Block *b : validPlacements) {
        if (!entryDistance.count(b))
          continue;
        if (entryDistance[b] < minBBdist) {
          minBBdist = entryDistance[b];
          minBB = b;
        }
      }

      if (!minBB || minBB == op.getBlock())
        continue;

      auto prev = std::prev(iter);
      op.moveBefore(minBB->getTerminator());
      iter = prev;
    }

    // Add successors of this block
    for (Block *succ : block->getSuccessors()) {
      if (!workDone.count(succ)) {
        workPending.insert(succ);
        assert(entryDistance.count(block) &&
               "ECM: block has to be present in entryDistance map");
        unsigned nextDist = entryDistance[block] + 1;
        entryDistance.insert(std::make_pair(succ, nextDist));
      }
    }
  }
}

std::unique_ptr<OperationPass<llhd::ProcOp>>
circt::llhd::createEarlyCodeMotionPass() {
  return std::make_unique<EarlyCodeMotionPass>();
}
