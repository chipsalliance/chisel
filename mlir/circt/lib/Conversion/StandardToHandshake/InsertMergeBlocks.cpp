//===- InsertMergeBlocks.cpp - Insert Merge Blocks --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "circt/Conversion/StandardToHandshake.h"
#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt;

/// Replaces the branching to oldDest of with an equivalent operation that
/// instead branches to newDest.
static LogicalResult changeBranchTarget(Block *block, Block *oldDest,
                                        Block *newDest,
                                        ConversionPatternRewriter &rewriter) {
  rewriter.setInsertionPointToEnd(block);
  auto term = block->getTerminator();
  return llvm::TypeSwitch<Operation *, LogicalResult>(term)
      .Case<cf::BranchOp>([&](auto branchOp) {
        rewriter.replaceOpWithNewOp<cf::BranchOp>(branchOp, newDest,
                                                  branchOp->getOperands());
        return success();
      })
      .Case<cf::CondBranchOp>([&](auto condBr) {
        auto cond = condBr.getCondition();

        Block *trueDest = condBr.getTrueDest();
        Block *falseDest = condBr.getFalseDest();

        // Change to the correct destination.
        if (trueDest == oldDest)
          trueDest = newDest;

        if (falseDest == oldDest)
          falseDest = newDest;

        rewriter.replaceOpWithNewOp<cf::CondBranchOp>(
            condBr, cond, trueDest, condBr.getTrueOperands(), falseDest,
            condBr.getFalseOperands());
        return success();
      })
      .Default([&](Operation *op) {
        return op->emitError("Unexpected terminator that cannot be handled.");
      });
}

/// Creates a new intermediate block that b1 and b2 branch to. The new block
/// branches to their common successor oldSucc.
static FailureOr<Block *> buildMergeBlock(Block *b1, Block *b2, Block *oldSucc,
                                          ConversionPatternRewriter &rewriter) {
  auto blockArgTypes = oldSucc->getArgumentTypes();
  SmallVector<Location> argLocs(blockArgTypes.size(), rewriter.getUnknownLoc());

  Block *res = rewriter.createBlock(oldSucc, blockArgTypes, argLocs);
  rewriter.create<cf::BranchOp>(rewriter.getUnknownLoc(), oldSucc,
                                res->getArguments());

  if (failed(changeBranchTarget(b1, oldSucc, res, rewriter)))
    return failure();
  if (failed(changeBranchTarget(b2, oldSucc, res, rewriter)))
    return failure();

  return res;
}

namespace {
/// A dual CFG that contracts cycles into single logical blocks.
struct DualGraph {
  DualGraph(Region &r, CFGLoopInfo &loopInfo);

  size_t getNumPredecessors(Block *b) { return predCnts.lookup(b); }
  void getPredecessors(Block *b, SmallVectorImpl<Block *> &res);

  size_t getNumSuccessors(Block *b) { return succMap.lookup(b).size(); }
  ArrayRef<Block *> getSuccessors(Block *b) {
    return succMap.find(b)->getSecond();
  }

  // If the block is part of a contracted block, the header of the contracted
  // block is returned. Otherwise, the block itself is returned.
  Block *lookupDualBlock(Block *b);
  DenseMap<Block *, size_t> getPredCountMapCopy() { return predCnts; }

private:
  CFGLoopInfo &loopInfo;

  DenseMap<Block *, SmallVector<Block *>> succMap;
  DenseMap<Block *, size_t> predCnts;
};
} // namespace

DualGraph::DualGraph(Region &r, CFGLoopInfo &loopInfo)
    : loopInfo(loopInfo), succMap(), predCnts() {
  for (Block &b : r) {
    CFGLoop *loop = loopInfo.getLoopFor(&b);

    if (loop && loop->getHeader() != &b)
      continue;

    // Create and get a new succ map entry for the current block.
    SmallVector<Block *> &succs =
        succMap.try_emplace(&b, SmallVector<Block *>()).first->getSecond();

    // NOTE: This assumes that there is only one exiting node, i.e., not
    // two blocks from the same loop can be predecessors of one block.
    unsigned predCnt = 0;
    for (auto *pred : b.getPredecessors())
      if (!loop || !loop->contains(pred))
        predCnt++;

    if (loop && loop->getHeader() == &b)
      loop->getExitBlocks(succs);
    else
      llvm::copy(b.getSuccessors(), std::back_inserter(succs));

    predCnts.try_emplace(&b, predCnt);
  }
}

Block *DualGraph::lookupDualBlock(Block *b) {
  CFGLoop *loop = loopInfo.getLoopFor(b);
  if (!loop)
    return b;

  return loop->getHeader();
}

void DualGraph::getPredecessors(Block *b, SmallVectorImpl<Block *> &res) {
  CFGLoop *loop = loopInfo.getLoopFor(b);
  assert((!loop || loop->getHeader() == b) &&
         "can only get predecessors of blocks in the graph");

  for (auto *pred : b->getPredecessors()) {
    if (loop && loop->contains(pred))
      continue;

    if (CFGLoop *predLoop = loopInfo.getLoopFor(pred)) {
      assert(predLoop->getExitBlock() &&
             "multiple exit blocks are not yet supported");
      res.push_back(predLoop->getHeader());
      continue;
    }
    res.push_back(pred);
  }
}

namespace {
using BlockToBlockMap = DenseMap<Block *, Block *>;
/// A helper class to store the split block information gathered during analysis
/// of the CFG.
struct SplitInfo {
  /// Points to the last split block that dominates the block.
  BlockToBlockMap in;
  /// Either points to the last split block or to itself, if the block itself is
  /// a split block.
  BlockToBlockMap out;
};
} // namespace

/// Builds a binary merge block tree for the predecessors of currBlock.
static LogicalResult buildMergeBlocks(Block *currBlock, SplitInfo &splitInfo,
                                      Block *predDom,
                                      ConversionPatternRewriter &rewriter,
                                      DualGraph &graph) {
  SmallVector<Block *> preds;
  llvm::copy(currBlock->getPredecessors(), std::back_inserter(preds));

  // Map from split blocks to blocks that descend from it.
  DenseMap<Block *, Block *> predsToConsider;

  while (!preds.empty()) {
    Block *pred = preds.pop_back_val();
    Block *splitBlock = splitInfo.out.lookup(graph.lookupDualBlock(pred));
    if (splitBlock == predDom)
      // Needs no additional merge block, as this directly descends from the
      // correct split block.
      continue;

    if (predsToConsider.count(splitBlock) == 0) {
      // No other block with the same split block was found yet, so just store
      // it and wait for a match.
      predsToConsider.try_emplace(splitBlock, pred);
      continue;
    }

    // Found a pair, so insert a new merge block for them.
    Block *other = predsToConsider.lookup(splitBlock);
    predsToConsider.erase(splitBlock);

    FailureOr<Block *> mergeBlock =
        buildMergeBlock(pred, other, currBlock, rewriter);
    if (failed(mergeBlock))
      return failure();

    // Update info for the newly created block.
    Block *splitIn = splitInfo.in.lookup(splitBlock);
    splitInfo.in.try_emplace(*mergeBlock, splitIn);
    // By construction, this block has only one successor, therefore, out == in.
    splitInfo.out.try_emplace(*mergeBlock, splitIn);

    preds.push_back(*mergeBlock);
  }
  if (!predsToConsider.empty())
    return currBlock->getParentOp()->emitError(
        "irregular control flow is not yet supported");
  return success();
}

/// Checks preconditions of this transformation.
static LogicalResult preconditionCheck(Region &r, CFGLoopInfo &loopInfo) {
  for (auto &info : loopInfo.getTopLevelLoops())
    // Does only return a block if it is the only exit block.
    if (!info->getExitBlock())
      return r.getParentOp()->emitError(
          "multiple exit blocks are not yet supported");

  return success();
}

/// Insert additional blocks that serve as counterparts to the blocks that
/// diverged the control flow.
/// The resulting merge block tree is guaranteed to be a binary tree.
///
/// This transformation does not affect any blocks that are part of a loop as it
/// treats a loop as one logical block.
/// Irregular control flow is not supported and results in a failed
/// transformation.
LogicalResult circt::insertMergeBlocks(Region &r,
                                       ConversionPatternRewriter &rewriter) {
  Block *entry = &r.front();
  DominanceInfo domInfo(r.getParentOp());

  CFGLoopInfo loopInfo(domInfo.getDomTree(&r));
  if (failed(preconditionCheck(r, loopInfo)))
    return failure();

  // Traversing the graph in topological order can be simply done with a stack.
  SmallVector<Block *> stack;
  stack.push_back(entry);

  // Holds the graph that contains the relevant blocks. It for example contracts
  // loops into one block to preserve a DAG structure.
  DualGraph graph(r, loopInfo);

  // Counts the amount of predecessors remaining.
  auto predsToVisit = graph.getPredCountMapCopy();

  SplitInfo splitInfo;

  while (!stack.empty()) {
    Block *currBlock = stack.pop_back_val();

    Block *in = nullptr;
    Block *out = nullptr;

    bool isMergeBlock = graph.getNumPredecessors(currBlock) > 1;
    bool isSplitBlock = graph.getNumSuccessors(currBlock) > 1;

    SmallVector<Block *> preds;
    graph.getPredecessors(currBlock, preds);

    if (isMergeBlock) {
      Block *predDom = currBlock;
      for (auto *pred : preds) {
        predDom = domInfo.findNearestCommonDominator(predDom, pred);
      }

      if (failed(
              buildMergeBlocks(currBlock, splitInfo, predDom, rewriter, graph)))
        return failure();

      // The sub-CFG created by the predDom (split block) and the current merge
      // block can logically be treated like a single block, thus their "in"s
      // are the same.
      in = splitInfo.in.lookup(predDom);
    } else if (!preds.empty()) {
      Block *pred = preds.front();

      in = splitInfo.out.lookup(pred);
    }

    if (isSplitBlock)
      out = currBlock;
    else
      out = in;

    splitInfo.in.try_emplace(currBlock, in);
    splitInfo.out.try_emplace(currBlock, out);

    for (auto *succ : graph.getSuccessors(currBlock)) {
      auto it = predsToVisit.find(succ);
      unsigned predsRemaining = --(it->getSecond());
      // Pushing the block on the stack once all it's successors were visited
      // ensures a topological traversal.
      if (predsRemaining == 0)
        stack.push_back(succ);
    }
  }

  return success();
}

namespace {

using PtrSet = SmallPtrSet<Operation *, 4>;

struct FuncOpPattern : public OpConversionPattern<func::FuncOp> {

  FuncOpPattern(PtrSet &rewrittenFuncs, MLIRContext *ctx)
      : OpConversionPattern(ctx), rewrittenFuncs(rewrittenFuncs) {}

  LogicalResult
  matchAndRewrite(func::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.startRootUpdate(op);

    if (!op.isExternal())
      if (failed(insertMergeBlocks(op.getRegion(), rewriter)))
        return failure();

    rewriter.finalizeRootUpdate(op);
    rewrittenFuncs.insert(op);

    return success();
  }

private:
  PtrSet &rewrittenFuncs;
};

struct InsertMergeBlocksPass
    : public InsertMergeBlocksBase<InsertMergeBlocksPass> {
public:
  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    // Remembers traversed functions to only apply the conversion once.
    PtrSet rewrittenFuncs;
    patterns.add<FuncOpPattern>(rewrittenFuncs, ctx);

    ConversionTarget target(*ctx);
    target.addDynamicallyLegalOp<func::FuncOp>(
        [&](func::FuncOp func) { return rewrittenFuncs.contains(func); });
    target.addLegalDialect<cf::ControlFlowDialect>();

    if (applyPartialConversion(getOperation(), target, std::move(patterns))
            .failed())
      signalPassFailure();
  }
};

} // namespace

namespace circt {
std::unique_ptr<mlir::Pass> createInsertMergeBlocksPass() {
  return std::make_unique<InsertMergeBlocksPass>();
}
} // namespace circt
