//===- MaximizeSSA.cpp - SSA Maximization Pass ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the SSA maximization pass as well as utilities
// for converting a function with standard control flow into maximal SSA form.
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "circt/Conversion/StandardToHandshake.h"
#include "circt/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace circt;

static Block *getDefiningBlock(Value value) {
  // Value is either a block argument...
  if (auto blockArg = dyn_cast<BlockArgument>(value); blockArg)
    return blockArg.getParentBlock();

  // ... or an operation's result
  auto *defOp = value.getDefiningOp();
  assert(defOp);
  return defOp->getBlock();
}

static LogicalResult addArgToTerminator(Block *block, Block *predBlock,
                                        Value value) {

  // Identify terminator branching instruction in predecessor block
  auto branchOp = dyn_cast<BranchOpInterface>(predBlock->getTerminator());
  if (!branchOp)
    return predBlock->getTerminator()->emitError(
        "Expected terminator operation of block to be a "
        "branch-like operation");

  // In the predecessor block's terminator, find all successors that equal
  // the block and add the value to the list of operands it's passed
  for (auto [idx, succBlock] : llvm::enumerate(branchOp->getSuccessors()))
    if (succBlock == block)
      branchOp.getSuccessorOperands(idx).append(value);

  return success();
}

bool circt::isRegionSSAMaximized(Region &region) {

  // Check whether all operands used within each block are also defined within
  // the same block
  for (auto &block : region.getBlocks())
    for (auto &op : block.getOperations())
      for (auto operand : op.getOperands())
        if (getDefiningBlock(operand) != &block)
          return false;

  return true;
}

bool circt::SSAMaximizationStrategy::maximizeBlock(Block *block) {
  return true;
}
bool circt::SSAMaximizationStrategy::maximizeArgument(BlockArgument arg) {
  return true;
}
bool circt::SSAMaximizationStrategy::maximizeOp(Operation *op) { return true; }
bool circt::SSAMaximizationStrategy::maximizeResult(OpResult res) {
  return true;
}

LogicalResult circt::maximizeSSA(Value value, PatternRewriter &rewriter) {

  // Identify the basic block in which the value is defined
  Block *defBlock = getDefiningBlock(value);

  // Identify all basic blocks in which the value is used (excluding the
  // value-defining block)
  DenseSet<Block *> blocksUsing;
  for (auto &use : value.getUses()) {
    auto *block = use.getOwner()->getBlock();
    if (block != defBlock)
      blocksUsing.insert(block);
  }

  // Prepare a stack to iterate over the list of basic blocks that must be
  // modified for the value to be in maximum SSA form. At all points,
  // blocksUsing is a non-strict superset of the elements contained in
  // blocksToVisit
  SmallVector<Block *> blocksToVisit(blocksUsing.begin(), blocksUsing.end());

  // Backtrack from all blocks using the value to the value-defining basic
  // block, adding a new block argument for the value along the way. Keep
  // track of which blocks have already been modified to avoid visiting a
  // block more than once while backtracking (possible due to branching
  // control flow)
  DenseMap<Block *, BlockArgument> blockToArg;
  while (!blocksToVisit.empty()) {
    // Pop the basic block at the top of the stack
    auto *block = blocksToVisit.pop_back_val();

    // Add an argument to the block to hold the value
    blockToArg[block] =
        block->addArgument(value.getType(), rewriter.getUnknownLoc());

    // In all unique block predecessors, modify the terminator branching
    // instruction to also pass the value to the block
    SmallPtrSet<Block *, 8> uniquePredecessors;
    for (auto *predBlock : block->getPredecessors()) {
      // If we have already visited the block predecessor, skip it. It's
      // possible to get duplicate block predecessors when there exists a
      // conditional branch with both branches going to the same block e.g.,
      // cf.cond_br %cond, ^bbx, ^bbx
      if (auto [_, newPredecessor] = uniquePredecessors.insert(predBlock);
          !newPredecessor) {
        continue;
      }

      // Modify the terminator instruction
      if (failed(addArgToTerminator(block, predBlock, value)))
        return failure();

      // Now the predecessor block is using the value, so we must also make sure
      // to visit it
      if (predBlock != defBlock)
        if (auto [_, blockNewlyUsing] = blocksUsing.insert(predBlock);
            blockNewlyUsing)
          blocksToVisit.push_back(predBlock);
    }
  }

  // Replace all uses of the value with the newly added block arguments
  SmallVector<Operation *> users;
  for (auto &use : value.getUses()) {
    auto *owner = use.getOwner();
    if (owner->getBlock() != defBlock)
      users.push_back(owner);
  }
  for (auto *user : users)
    user->replaceUsesOfWith(value, blockToArg[user->getBlock()]);

  return success();
}

LogicalResult circt::maximizeSSA(Operation *op,
                                 SSAMaximizationStrategy &strategy,
                                 PatternRewriter &rewriter) {
  // Apply SSA maximization on each of the operation's results
  for (auto res : op->getResults())
    if (strategy.maximizeResult(res))
      if (failed(maximizeSSA(res, rewriter)))
        return failure();

  return success();
}

LogicalResult circt::maximizeSSA(Block *block,
                                 SSAMaximizationStrategy &strategy,
                                 PatternRewriter &rewriter) {
  // Apply SSA maximization on each of the block's arguments
  for (auto arg : block->getArguments())
    if (strategy.maximizeArgument(arg))
      if (failed(maximizeSSA(arg, rewriter)))
        return failure();

  // Apply SSA maximization on each of the block's operations
  for (auto &op : block->getOperations())
    if (strategy.maximizeOp(&op))
      if (failed(maximizeSSA(&op, strategy, rewriter)))
        return failure();

  return success();
}

LogicalResult circt::maximizeSSA(Region &region,
                                 SSAMaximizationStrategy &strategy,
                                 PatternRewriter &rewriter) {
  // Apply SSA maximization on each of the region's block
  for (auto &block : region.getBlocks())
    if (strategy.maximizeBlock(&block))
      if (failed(maximizeSSA(&block, strategy, rewriter)))
        return failure();

  return success();
}

namespace {

struct FuncOpMaxSSAConversion : public OpConversionPattern<func::FuncOp> {

  FuncOpMaxSSAConversion(MLIRContext *ctx) : OpConversionPattern(ctx) {}

  LogicalResult
  matchAndRewrite(func::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LogicalResult conversionStatus = success();
    rewriter.updateRootInPlace(op, [&] {
      SSAMaximizationStrategy strategy;
      if (failed(maximizeSSA(op.getRegion(), strategy, rewriter)))
        conversionStatus = failure();
    });
    return conversionStatus;
  }
};

struct MaximizeSSAPass : public MaximizeSSABase<MaximizeSSAPass> {
public:
  void runOnOperation() override {
    auto *ctx = &getContext();

    RewritePatternSet patterns{ctx};
    patterns.add<FuncOpMaxSSAConversion>(ctx);
    ConversionTarget target(*ctx);

    // Check that the function is correctly SSA-maximized after the pattern has
    // been applied
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp func) {
      return isRegionSSAMaximized(func.getBody());
    });

    // Each function in the module is turned into maximal SSA form
    // independently of the others. Function signatures are never modified
    // by SSA maximization
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace circt {
std::unique_ptr<mlir::Pass> createMaximizeSSAPass() {
  return std::make_unique<MaximizeSSAPass>();
}
} // namespace circt
