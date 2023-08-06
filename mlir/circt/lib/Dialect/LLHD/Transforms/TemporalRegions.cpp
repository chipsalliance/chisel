//===- TemporalRegions.cpp - LLHD temporal regions analysis ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements an Analysis for Behavioral LLHD to find the temporal
// regions of an LLHD process.
//
//===----------------------------------------------------------------------===//

#include "TemporalRegions.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace circt;

static void addBlockToTR(Block *block, int tr, DenseMap<Block *, int> &blockMap,
                         DenseMap<int, SmallVector<Block *, 8>> &trMap) {
  blockMap.insert(std::make_pair(block, tr));
  SmallVector<Block *, 8> b;
  b.push_back(block);
  trMap.insert(std::make_pair(tr, b));
}

static bool anyPredecessorHasWait(Block *block) {
  return std::any_of(block->pred_begin(), block->pred_end(), [](Block *pred) {
    return isa<llhd::WaitOp>(pred->getTerminator());
  });
}

static bool allPredecessorTRsKnown(Block *block,
                                   SmallPtrSetImpl<Block *> &known) {
  return std::all_of(block->pred_begin(), block->pred_end(), [&](Block *pred) {
    return std::find(known.begin(), known.end(), pred) != known.end();
  });
}

void llhd::TemporalRegionAnalysis::recalculate(Operation *operation) {
  assert(isa<ProcOp>(operation) &&
         "TemporalRegionAnalysis: operation needs to be llhd::ProcOp");
  ProcOp proc = cast<ProcOp>(operation);
  int nextTRnum = -1;
  blockMap.clear();
  trMap.clear();

  SmallPtrSet<Block *, 32> workQueue;
  SmallPtrSet<Block *, 32> workDone;

  // Add the entry block and all blocks targeted by a wait terminator to the
  // initial work queue because they are always the entry block of a new TR
  workQueue.insert(&proc.getBody().front());
  proc.walk([&](WaitOp wait) { workQueue.insert(wait.getDest()); });

  while (!workQueue.empty()) {
    // Find basic block in the work queue which has all predecessors already
    // processed or at least one predecessor has a wait terminator
    auto iter =
        std::find_if(workQueue.begin(), workQueue.end(), [&](Block *block) {
          return allPredecessorTRsKnown(block, workDone) ||
                 anyPredecessorHasWait(block);
        });

    // If no element in the work queue has all predecessors finished or is
    // targeted by a wait, there is probably a loop within a TR, in this case we
    // are conservatively assign a new temporal region
    if (iter == workQueue.end())
      iter = workQueue.begin();

    // This is the current block to be processed
    Block *block = *iter;

    // Delete this block from the work queue and add it to the list of alredy
    // processed blocks
    workQueue.erase(block);
    workDone.insert(block);

    // Add all successors of this block which were not already processed to the
    // work queue
    for (Block *succ : block->getSuccessors()) {
      if (!workDone.count(succ))
        workQueue.insert(succ);
    }

    // The entry block is always assigned -1 as a placeholder as this block must
    // not contain any temporal operations
    if (block->isEntryBlock()) {
      addBlockToTR(block, -1, blockMap, trMap);
      // If at least one predecessor has a wait terminator or at least one
      // predecessor has an unknown temporal region or not all predecessors have
      // the same TR, create a new TR
    } else if (!allPredecessorTRsKnown(block, workDone) ||
               anyPredecessorHasWait(block) ||
               !(std::adjacent_find(block->pred_begin(), block->pred_end(),
                                    [&](Block *pred1, Block *pred2) {
                                      return blockMap[pred1] != blockMap[pred2];
                                    }) == block->pred_end())) {
      addBlockToTR(block, ++nextTRnum, blockMap, trMap);
      // If all predecessors have the same TR and none has a wait terminator,
      // inherit the TR
    } else {
      int tr = blockMap[*block->pred_begin()];
      blockMap.insert(std::make_pair(block, tr));
      trMap[tr].push_back(block);
      trMap[tr].erase(std::unique(trMap[tr].begin(), trMap[tr].end()),
                      trMap[tr].end());
    }
  }

  numTRs = nextTRnum + 1;
}

int llhd::TemporalRegionAnalysis::getBlockTR(Block *block) {
  assert(blockMap.count(block) &&
         "This block is not present in the temporal regions map.");
  return blockMap[block];
}

SmallVector<Block *, 8> llhd::TemporalRegionAnalysis::getBlocksInTR(int tr) {
  if (!trMap.count(tr))
    return SmallVector<Block *, 8>();
  return trMap[tr];
}

SmallVector<Block *, 8>
llhd::TemporalRegionAnalysis::getExitingBlocksInTR(int tr) {
  SmallVector<Block *, 8> blocks = getBlocksInTR(tr);
  SmallVector<Block *, 8> exitingBlocks;
  for (Block *block : blocks) {
    for (auto succ : block->getSuccessors()) {
      if (blockMap[succ] != blockMap[block] ||
          isa<WaitOp>(block->getTerminator())) {
        exitingBlocks.push_back(block);
        break;
      }
    }
  }
  return exitingBlocks;
}

SmallVector<int, 8> llhd::TemporalRegionAnalysis::getTRSuccessors(int tr) {
  SmallVector<int, 8> res;
  for (Block *block : getExitingBlocksInTR(tr)) {
    for (Block *succ : block->getSuccessors()) {
      if (getBlockTR(succ) != tr || isa<llhd::WaitOp>(block->getTerminator()))
        res.push_back(getBlockTR(succ));
    }
  }
  res.erase(std::unique(res.begin(), res.end()), res.end());
  return res;
}

Block *llhd::TemporalRegionAnalysis::getTREntryBlock(int tr) {
  for (Block *block : getBlocksInTR(tr)) {
    if (block->hasNoPredecessors() || anyPredecessorHasWait(block))
      return block;

    for (Block *pred : block->getPredecessors()) {
      if (getBlockTR(pred) != tr)
        return block;
    }
  }
  return nullptr;
}
