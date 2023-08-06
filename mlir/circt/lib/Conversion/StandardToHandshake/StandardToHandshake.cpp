//===- StandardToHandshake.cpp - Convert standard MLIR into dataflow IR ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This is the main Standard to Handshake Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/StandardToHandshake.h"
#include "../PassDetail.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "circt/Support/BackedgeBuilder.h"
#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include <list>
#include <map>

using namespace mlir;
using namespace mlir::func;
using namespace mlir::affine;
using namespace circt;
using namespace circt::handshake;
using namespace std;

// ============================================================================
// Partial lowering infrastructure
// ============================================================================

namespace {
template <typename TOp>
class LowerOpTarget : public ConversionTarget {
public:
  explicit LowerOpTarget(MLIRContext &context) : ConversionTarget(context) {
    loweredOps.clear();
    addLegalDialect<HandshakeDialect>();
    addLegalDialect<mlir::func::FuncDialect>();
    addLegalDialect<mlir::arith::ArithDialect>();
    addIllegalDialect<mlir::scf::SCFDialect>();
    addIllegalDialect<AffineDialect>();

    /// The root operation to be replaced is marked dynamically legal
    /// based on the lowering status of the given operation, see
    /// PartialLowerOp.
    addDynamicallyLegalOp<TOp>([&](const auto &op) { return loweredOps[op]; });
  }
  DenseMap<Operation *, bool> loweredOps;
};

/// Default function for partial lowering of handshake::FuncOp. Lowering is
/// achieved by a provided partial lowering function.
///
/// A partial lowering function may only replace a subset of the operations
/// within the funcOp currently being lowered. However, the dialect conversion
/// scheme requires the matched root operation to be replaced/updated, if the
/// match was successful. To facilitate this, rewriter.updateRootInPlace
/// wraps the partial update function.
/// Next, the function operation is expected to go from illegal to legalized,
/// after matchAndRewrite returned true. To work around this,
/// LowerFuncOpTarget::loweredFuncs is used to communicate between the target
/// and the conversion, to indicate that the partial lowering was completed.
template <typename TOp>
struct PartialLowerOp : public ConversionPattern {
  using PartialLoweringFunc =
      std::function<LogicalResult(TOp, ConversionPatternRewriter &)>;

public:
  PartialLowerOp(LowerOpTarget<TOp> &target, MLIRContext *context,
                 LogicalResult &loweringResRef, const PartialLoweringFunc &fun)
      : ConversionPattern(TOp::getOperationName(), 1, context), target(target),
        loweringRes(loweringResRef), fun(fun) {}
  using ConversionPattern::ConversionPattern;
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    assert(isa<TOp>(op));
    rewriter.updateRootInPlace(
        op, [&] { loweringRes = fun(dyn_cast<TOp>(op), rewriter); });
    target.loweredOps[op] = true;
    return loweringRes;
  };

private:
  LowerOpTarget<TOp> &target;
  LogicalResult &loweringRes;
  // NOTE: this is basically the rewrite function
  PartialLoweringFunc fun;
};
} // namespace

// Convenience function for running lowerToHandshake with a partial
// handshake::FuncOp lowering function.
template <typename TOp>
static LogicalResult partiallyLowerOp(
    const std::function<LogicalResult(TOp, ConversionPatternRewriter &)>
        &loweringFunc,
    MLIRContext *ctx, TOp op) {

  RewritePatternSet patterns(ctx);
  auto target = LowerOpTarget<TOp>(*ctx);
  LogicalResult partialLoweringSuccessfull = success();
  patterns.add<PartialLowerOp<TOp>>(target, ctx, partialLoweringSuccessfull,
                                    loweringFunc);
  return success(
      applyPartialConversion(op, target, std::move(patterns)).succeeded() &&
      partialLoweringSuccessfull.succeeded());
}

class LowerRegionTarget : public ConversionTarget {
public:
  explicit LowerRegionTarget(MLIRContext &context, Region &region)
      : ConversionTarget(context), region(region) {
    // The root operation is marked dynamically legal to ensure
    // the pattern on its region is only applied once.
    markUnknownOpDynamicallyLegal([&](Operation *op) {
      if (op != region.getParentOp())
        return true;
      return opLowered;
    });
  }
  bool opLowered = false;
  Region &region;
};

/// Allows to partially lower a region by matching on the parent operation to
/// then call the provided partial lowering function with the region and the
/// rewriter.
///
/// The interplay with the target is similar to PartialLowerOp
struct PartialLowerRegion : public ConversionPattern {
  using PartialLoweringFunc =
      std::function<LogicalResult(Region &, ConversionPatternRewriter &)>;

public:
  PartialLowerRegion(LowerRegionTarget &target, MLIRContext *context,
                     LogicalResult &loweringResRef,
                     const PartialLoweringFunc &fun)
      : ConversionPattern(target.region.getParentOp()->getName().getStringRef(),
                          1, context),
        target(target), loweringRes(loweringResRef), fun(fun) {}
  using ConversionPattern::ConversionPattern;
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.updateRootInPlace(
        op, [&] { loweringRes = fun(target.region, rewriter); });

    target.opLowered = true;
    return loweringRes;
  };

private:
  LowerRegionTarget &target;
  LogicalResult &loweringRes;
  PartialLoweringFunc fun;
};

LogicalResult
handshake::partiallyLowerRegion(const RegionLoweringFunc &loweringFunc,
                                MLIRContext *ctx, Region &r) {

  Operation *op = r.getParentOp();
  RewritePatternSet patterns(ctx);
  auto target = LowerRegionTarget(*ctx, r);
  LogicalResult partialLoweringSuccessfull = success();
  patterns.add<PartialLowerRegion>(target, ctx, partialLoweringSuccessfull,
                                   loweringFunc);
  return success(
      applyPartialConversion(op, target, std::move(patterns)).succeeded() &&
      partialLoweringSuccessfull.succeeded());
}

#define returnOnError(logicalResult)                                           \
  if (failed(logicalResult))                                                   \
    return failure();

// ============================================================================
// Start of lowering passes
// ============================================================================

Value HandshakeLowering::getBlockEntryControl(Block *block) const {
  auto it = blockEntryControlMap.find(block);
  assert(it != blockEntryControlMap.end() &&
         "No block entry control value registerred for this block!");
  return it->second;
}

void HandshakeLowering::setBlockEntryControl(Block *block, Value v) {
  blockEntryControlMap[block] = v;
}

void handshake::removeBasicBlocks(Region &r) {
  auto &entryBlock = r.front().getOperations();

  // Now that basic blocks are going to be removed, we can erase all cf-dialect
  // branches, and move ReturnOp to the entry block's end
  for (auto &block : r) {
    Operation &termOp = block.back();
    if (isa<mlir::cf::CondBranchOp, mlir::cf::BranchOp>(termOp))
      termOp.erase();
    else if (isa<handshake::ReturnOp>(termOp))
      entryBlock.splice(entryBlock.end(), block.getOperations(), termOp);
  }

  // Move all operations to entry block and erase other blocks.
  for (auto &block : llvm::make_early_inc_range(llvm::drop_begin(r, 1))) {
    entryBlock.splice(--entryBlock.end(), block.getOperations());
  }
  for (auto &block : llvm::make_early_inc_range(llvm::drop_begin(r, 1))) {
    block.clear();
    block.dropAllDefinedValueUses();
    for (size_t i = 0; i < block.getNumArguments(); i++) {
      block.eraseArgument(i);
    }
    block.erase();
  }
}

void removeBasicBlocks(handshake::FuncOp funcOp) {
  if (funcOp.isExternal())
    return; // nothing to do, external funcOp.

  removeBasicBlocks(funcOp.getBody());
}

static LogicalResult isValidMemrefType(Location loc, mlir::MemRefType type) {
  if (type.getNumDynamicDims() != 0 || type.getShape().size() != 1)
    return emitError(loc) << "memref's must be both statically sized and "
                             "unidimensional.";
  return success();
}

static unsigned getBlockPredecessorCount(Block *block) {
  // Returns number of block predecessors
  auto predecessors = block->getPredecessors();
  return std::distance(predecessors.begin(), predecessors.end());
}

// Insert appropriate type of Merge CMerge for control-only path,
// Merge for single-successor blocks, Mux otherwise
HandshakeLowering::MergeOpInfo
HandshakeLowering::insertMerge(Block *block, Value val,
                               BackedgeBuilder &edgeBuilder,
                               ConversionPatternRewriter &rewriter) {
  unsigned numPredecessors = getBlockPredecessorCount(block);
  auto insertLoc = block->front().getLoc();
  SmallVector<Backedge> dataEdges;
  SmallVector<Value> operands;

  // Every block (except the entry block) needs to feed it's entry control into
  // a control merge
  if (val == getBlockEntryControl(block)) {

    Operation *mergeOp;
    if (block == &r.front()) {
      // For consistency within the entry block, replace the latter's entry
      // control with the output of a MergeOp that takes the control-only
      // network's start point as input. This makes it so that only the
      // MergeOp's output is used as a control within the entry block, instead
      // of a combination of the MergeOp's output and the function/block control
      // argument. Taking this step out should have no impact on functionality
      // but would make the resulting IR less "regular"
      operands.push_back(val);
      mergeOp = rewriter.create<handshake::MergeOp>(insertLoc, operands);
    } else {
      for (unsigned i = 0; i < numPredecessors; i++) {
        auto edge = edgeBuilder.get(rewriter.getNoneType());
        dataEdges.push_back(edge);
        operands.push_back(Value(edge));
      }
      mergeOp = rewriter.create<handshake::ControlMergeOp>(insertLoc, operands);
    }
    setBlockEntryControl(block, mergeOp->getResult(0));
    return MergeOpInfo{mergeOp, val, dataEdges};
  }

  // Every live-in value to a block is passed through a merge-like operation,
  // even when it's not required for circuit correctness (useless merge-like
  // operations are removed down the line during handshake canonicalization)

  // Insert "dummy" MergeOp's for blocks with less than two predecessors
  if (numPredecessors <= 1) {
    if (numPredecessors == 0) {
      // All of the entry block's block arguments get passed through a dummy
      // MergeOp. There is no need for a backedge here as the unique operand can
      // be resolved immediately
      operands.push_back(val);
    } else {
      // The value incoming from the single block predecessor will be resolved
      // later during merge reconnection
      auto edge = edgeBuilder.get(val.getType());
      dataEdges.push_back(edge);
      operands.push_back(Value(edge));
    }
    auto merge = rewriter.create<handshake::MergeOp>(insertLoc, operands);
    return MergeOpInfo{merge, val, dataEdges};
  }

  // Create a backedge for the index operand, and another one for each data
  // operand. The index operand will eventually resolve to the current block's
  // control merge index output, while data operands will resolve to their
  // respective values from each block predecessor
  Backedge indexEdge = edgeBuilder.get(rewriter.getIndexType());
  for (unsigned i = 0; i < numPredecessors; i++) {
    auto edge = edgeBuilder.get(val.getType());
    dataEdges.push_back(edge);
    operands.push_back(Value(edge));
  }
  auto mux =
      rewriter.create<handshake::MuxOp>(insertLoc, Value(indexEdge), operands);
  return MergeOpInfo{mux, val, dataEdges, indexEdge};
}

HandshakeLowering::BlockOps
HandshakeLowering::insertMergeOps(HandshakeLowering::ValueMap &mergePairs,
                                  BackedgeBuilder &edgeBuilder,
                                  ConversionPatternRewriter &rewriter) {
  HandshakeLowering::BlockOps blockMerges;
  for (Block &block : r) {
    rewriter.setInsertionPointToStart(&block);

    // All of the block's live-ins are passed explictly through block arguments
    // thanks to prior SSA maximization
    for (auto &arg : block.getArguments()) {
      // No merges on memref block arguments; these are handled separately
      if (arg.getType().isa<mlir::MemRefType>())
        continue;

      auto mergeInfo = insertMerge(&block, arg, edgeBuilder, rewriter);
      blockMerges[&block].push_back(mergeInfo);
      mergePairs[arg] = mergeInfo.op->getResult(0);
    }
  }
  return blockMerges;
}

// Get value from predBlock which will be set as operand of op (merge)
static Value getMergeOperand(HandshakeLowering::MergeOpInfo mergeInfo,
                             Block *predBlock) {
  // The input value to the merge operations
  Value srcVal = mergeInfo.val;
  // The block the merge operation belongs to
  Block *block = mergeInfo.op->getBlock();

  // The block terminator is either a cf-level branch or cf-level conditional
  // branch. In either case, identify the value passed to the block using its
  // index in the list of block arguments
  unsigned index = srcVal.cast<BlockArgument>().getArgNumber();
  Operation *termOp = predBlock->getTerminator();
  if (mlir::cf::CondBranchOp br = dyn_cast<mlir::cf::CondBranchOp>(termOp)) {
    // Block should be one of the two destinations of the conditional branch
    if (block == br.getTrueDest())
      return br.getTrueOperand(index);
    assert(block == br.getFalseDest());
    return br.getFalseOperand(index);
  }
  if (isa<mlir::cf::BranchOp>(termOp))
    return termOp->getOperand(index);
  return nullptr;
}

static void removeBlockOperands(Region &f) {
  // Remove all block arguments, they are no longer used
  // eraseArguments also removes corresponding branch operands
  for (Block &block : f) {
    if (!block.isEntryBlock()) {
      int x = block.getNumArguments() - 1;
      for (int i = x; i >= 0; --i)
        block.eraseArgument(i);
    }
  }
}

/// Returns the first occurance of an operation of type TOp, else, returns
/// null op.
template <typename TOp>
static Operation *getFirstOp(Block *block) {
  auto ops = block->getOps<TOp>();
  if (ops.empty())
    return nullptr;
  return *ops.begin();
}

static Operation *getControlMerge(Block *block) {
  return getFirstOp<ControlMergeOp>(block);
}

static ConditionalBranchOp getControlCondBranch(Block *block) {
  for (auto cbranch : block->getOps<handshake::ConditionalBranchOp>()) {
    if (cbranch.isControl())
      return cbranch;
  }
  return nullptr;
}

static void reconnectMergeOps(Region &r,
                              HandshakeLowering::BlockOps blockMerges,
                              HandshakeLowering::ValueMap &mergePairs) {
  // At this point all merge-like operations have backedges as operands.
  // We here replace all backedge values with appropriate value from
  // predecessor block. The predecessor can either be a merge, the original
  // defining value, or a branch operand.

  for (Block &block : r) {
    for (auto &mergeInfo : blockMerges[&block]) {
      int operandIdx = 0;
      // Set appropriate operand from each predecessor block
      for (auto *predBlock : block.getPredecessors()) {
        Value mgOperand = getMergeOperand(mergeInfo, predBlock);
        assert(mgOperand != nullptr);
        if (!mgOperand.getDefiningOp()) {
          assert(mergePairs.count(mgOperand));
          mgOperand = mergePairs[mgOperand];
        }
        mergeInfo.dataEdges[operandIdx].setValue(mgOperand);
        operandIdx++;
      }

      // Reconnect all operands originating from livein defining value through
      // corresponding merge of that block
      for (Operation &opp : block)
        if (!isa<MergeLikeOpInterface>(opp))
          opp.replaceUsesOfWith(mergeInfo.val, mergeInfo.op->getResult(0));
    }
  }

  // Connect select operand of muxes to control merge's index result in all
  // blocks with more than one predecessor
  for (Block &block : r) {
    if (getBlockPredecessorCount(&block) > 1) {
      Operation *cntrlMg = getControlMerge(&block);
      assert(cntrlMg != nullptr);

      for (auto &mergeInfo : blockMerges[&block]) {
        if (mergeInfo.op != cntrlMg) {
          // If the block has multiple predecessors, merge-like operation that
          // are not the block's control merge must have an index operand (at
          // this point, an index backedge)
          assert(mergeInfo.indexEdge.has_value());
          (*mergeInfo.indexEdge).setValue(cntrlMg->getResult(1));
        }
      }
    }
  }

  removeBlockOperands(r);
}

static bool isAllocOp(Operation *op) {
  return isa<memref::AllocOp, memref::AllocaOp>(op);
}

LogicalResult
HandshakeLowering::addMergeOps(ConversionPatternRewriter &rewriter) {

  // Stores mapping from each value that pass through a merge operation to the
  // first result of that merge operation
  ValueMap mergePairs;

  // Create backedge builder to manage operands of merge operations between
  // insertion and reconnection
  BackedgeBuilder edgeBuilder{rewriter, r.front().front().getLoc()};

  // Insert merge operations (with backedges instead of actual operands)
  BlockOps mergeOps = insertMergeOps(mergePairs, edgeBuilder, rewriter);

  // Reconnect merge operations with values incoming from predecessor blocks
  // and resolve all backedges that were created during merge insertion
  reconnectMergeOps(r, mergeOps, mergePairs);
  return success();
}

static bool isLiveOut(Value val) {
  // Identifies liveout values after adding Merges
  for (auto &u : val.getUses())
    // Result is liveout if used by some Merge block
    if (isa<MergeLikeOpInterface>(u.getOwner()))
      return true;
  return false;
}

// A value can have multiple branches in a single successor block
// (for instance, there can be an SSA phi and a merge that we insert)
// This function determines the number of branches to insert based on the
// value uses in successor blocks
static int getBranchCount(Value val, Block *block) {
  int uses = 0;
  for (int i = 0, e = block->getNumSuccessors(); i < e; ++i) {
    int curr = 0;
    Block *succ = block->getSuccessor(i);
    for (auto &u : val.getUses()) {
      if (u.getOwner()->getBlock() == succ)
        curr++;
    }
    uses = (curr > uses) ? curr : uses;
  }
  return uses;
}

namespace {

/// This class inserts a reorder prevention mechanism for blocks with multiple
/// successors. Such a mechanism is required to guarantee correct execution in a
/// multi-threaded usage of the circuits.
///
/// The order of the results matches the order of the traversals of the
/// divergence point. A FIFO buffer stores the condition of the conditional
/// branch. The buffer feeds a mux that guarantees the correct out-order.
class FeedForwardNetworkRewriter {
public:
  FeedForwardNetworkRewriter(HandshakeLowering &hl,
                             ConversionPatternRewriter &rewriter)
      : hl(hl), rewriter(rewriter), postDomInfo(hl.getRegion().getParentOp()),
        domInfo(hl.getRegion().getParentOp()),
        loopInfo(domInfo.getDomTree(&hl.getRegion())) {}
  LogicalResult apply();

private:
  HandshakeLowering &hl;
  ConversionPatternRewriter &rewriter;
  PostDominanceInfo postDomInfo;
  DominanceInfo domInfo;
  CFGLoopInfo loopInfo;

  using BlockPair = std::pair<Block *, Block *>;
  using BlockPairs = SmallVector<BlockPair>;
  LogicalResult findBlockPairs(BlockPairs &blockPairs);

  BufferOp buildSplitNetwork(Block *splitBlock,
                             handshake::ConditionalBranchOp &ctrlBr);
  LogicalResult buildMergeNetwork(Block *mergeBlock, BufferOp buf,
                                  handshake::ConditionalBranchOp &ctrlBr);

  // Determines if the cmerge inpus match the cond_br output order.
  bool requiresOperandFlip(ControlMergeOp &ctrlMerge,
                           handshake::ConditionalBranchOp &ctrlBr);
  bool formsIrreducibleCF(Block *splitBlock, Block *mergeBlock);
};
} // namespace

LogicalResult
HandshakeLowering::feedForwardRewriting(ConversionPatternRewriter &rewriter) {
  // Nothing to do on a single block region.
  if (this->getRegion().hasOneBlock())
    return success();
  return FeedForwardNetworkRewriter(*this, rewriter).apply();
}

static bool loopsHaveSingleExit(CFGLoopInfo &loopInfo) {
  for (CFGLoop *loop : loopInfo.getTopLevelLoops())
    if (!loop->getExitBlock())
      return false;
  return true;
}

bool FeedForwardNetworkRewriter::formsIrreducibleCF(Block *splitBlock,
                                                    Block *mergeBlock) {
  CFGLoop *loop = loopInfo.getLoopFor(mergeBlock);
  for (auto *mergePred : mergeBlock->getPredecessors()) {
    // Skip loop predecessors
    if (loop && loop->contains(mergePred))
      continue;

    // A DAG-CFG is irreducible, iff a merge block has a predecessor that can be
    // reached from both successors of a split node, e.g., neither is a
    // dominator.
    // => Their control flow can merge in other places, which makes this
    // irreducible.
    if (llvm::none_of(splitBlock->getSuccessors(), [&](Block *splitSucc) {
          if (splitSucc == mergeBlock || mergePred == splitBlock)
            return true;
          return domInfo.dominates(splitSucc, mergePred);
        }))
      return true;
  }
  return false;
}

static Operation *findBranchToBlock(Block *block) {
  Block *pred = *block->getPredecessors().begin();
  return pred->getTerminator();
}

LogicalResult
FeedForwardNetworkRewriter::findBlockPairs(BlockPairs &blockPairs) {
  // assumes that merge block insertion happended beforehand
  // Thus, for each split block, there exists one merge block which is the post
  // dominator of the child nodes.
  Region &r = hl.getRegion();
  Operation *parentOp = r.getParentOp();

  // Assumes that each loop has only one exit block. Such an error should
  // already be reported by the loop rewriting.
  assert(loopsHaveSingleExit(loopInfo) &&
         "expected loop to only have one exit block.");

  for (Block &b : r) {
    if (b.getNumSuccessors() < 2)
      continue;

    // Loop headers cannot be merge blocks.
    if (loopInfo.getLoopFor(&b))
      continue;

    assert(b.getNumSuccessors() == 2);
    Block *succ0 = b.getSuccessor(0);
    Block *succ1 = b.getSuccessor(1);

    if (succ0 == succ1)
      continue;

    Block *mergeBlock = postDomInfo.findNearestCommonDominator(succ0, succ1);

    // Precondition checks
    if (formsIrreducibleCF(&b, mergeBlock)) {
      return parentOp->emitError("expected only reducible control flow.")
                 .attachNote(findBranchToBlock(mergeBlock)->getLoc())
             << "This branch is involved in the irreducible control flow";
    }

    unsigned nonLoopPreds = 0;
    CFGLoop *loop = loopInfo.getLoopFor(mergeBlock);
    for (auto *pred : mergeBlock->getPredecessors()) {
      if (loop && loop->contains(pred))
        continue;
      nonLoopPreds++;
    }
    if (nonLoopPreds > 2)
      return parentOp
                 ->emitError("expected a merge block to have two predecessors. "
                             "Did you run the merge block insertion pass?")
                 .attachNote(findBranchToBlock(mergeBlock)->getLoc())
             << "This branch jumps to the illegal block";

    blockPairs.emplace_back(&b, mergeBlock);
  }

  return success();
}

LogicalResult FeedForwardNetworkRewriter::apply() {
  BlockPairs pairs;

  if (failed(findBlockPairs(pairs)))
    return failure();

  for (auto [splitBlock, mergeBlock] : pairs) {
    handshake::ConditionalBranchOp ctrlBr;
    BufferOp buffer = buildSplitNetwork(splitBlock, ctrlBr);
    if (failed(buildMergeNetwork(mergeBlock, buffer, ctrlBr)))
      return failure();
  }

  return success();
}

BufferOp FeedForwardNetworkRewriter::buildSplitNetwork(
    Block *splitBlock, handshake::ConditionalBranchOp &ctrlBr) {
  SmallVector<handshake::ConditionalBranchOp> branches;
  llvm::copy(splitBlock->getOps<handshake::ConditionalBranchOp>(),
             std::back_inserter(branches));

  auto *findRes = llvm::find_if(branches, [](auto br) {
    return br.getDataOperand().getType().template isa<NoneType>();
  });

  assert(findRes && "expected one branch for the ctrl signal");
  ctrlBr = *findRes;

  Value cond = ctrlBr.getConditionOperand();
  assert(llvm::all_of(branches, [&](auto branch) {
    return branch.getConditionOperand() == cond;
  }));

  Location loc = cond.getLoc();
  rewriter.setInsertionPointAfterValue(cond);

  // The buffer size defines the number of threads that can be concurently
  // traversing the sub-CFG starting at the splitBlock.
  size_t bufferSize = 2;
  // TODO how to size these?
  // Longest path in a CFG-DAG would be O(#blocks)

  return rewriter.create<handshake::BufferOp>(loc, cond, bufferSize,
                                              BufferTypeEnum::fifo);
}

LogicalResult FeedForwardNetworkRewriter::buildMergeNetwork(
    Block *mergeBlock, BufferOp buf, handshake::ConditionalBranchOp &ctrlBr) {
  // Replace control merge with mux
  auto ctrlMerges = mergeBlock->getOps<handshake::ControlMergeOp>();
  assert(std::distance(ctrlMerges.begin(), ctrlMerges.end()) == 1);

  handshake::ControlMergeOp ctrlMerge = *ctrlMerges.begin();
  // This input might contain irreducible loops that we cannot handle.
  if (ctrlMerge.getNumOperands() != 2)
    return ctrlMerge.emitError("expected cmerges to have two operands");
  rewriter.setInsertionPointAfter(ctrlMerge);
  Location loc = ctrlMerge->getLoc();

  // The newly inserted mux has to select the results from the correct operand.
  // As there is no guarantee on the order of cmerge inputs, the correct order
  // has to be determined first.
  bool requiresFlip = requiresOperandFlip(ctrlMerge, ctrlBr);
  SmallVector<Value> muxOperands;
  if (requiresFlip)
    muxOperands = llvm::to_vector(llvm::reverse(ctrlMerge.getOperands()));
  else
    muxOperands = llvm::to_vector(ctrlMerge.getOperands());

  Value newCtrl = rewriter.create<handshake::MuxOp>(loc, buf, muxOperands);

  Value cond = buf.getResult();
  if (requiresFlip) {
    // As the mux operand order is the flipped cmerge input order, the index
    // which replaces the output of the cmerge has to be flipped/negated as
    // well.
    cond = rewriter.create<arith::XOrIOp>(
        loc, cond.getType(), cond,
        rewriter.create<arith::ConstantOp>(
            loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 1)));
  }

  // Require a cast to index to stick to the type of the mux input.
  Value condAsIndex =
      rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), cond);

  hl.setBlockEntryControl(mergeBlock, newCtrl);

  // Replace with new ctrl value from mux and the index
  rewriter.replaceOp(ctrlMerge, {newCtrl, condAsIndex});
  return success();
}

bool FeedForwardNetworkRewriter::requiresOperandFlip(
    ControlMergeOp &ctrlMerge, handshake::ConditionalBranchOp &ctrlBr) {
  assert(ctrlMerge.getNumOperands() == 2 &&
         "Loops should already have been handled");

  Value fstOperand = ctrlMerge.getOperand(0);

  assert(ctrlBr.getTrueResult().hasOneUse() &&
         "expected the result of a branch to only have one user");
  Operation *trueUser = *ctrlBr.getTrueResult().user_begin();
  if (trueUser == ctrlBr)
    // The cmerge directly consumes the cond_br output.
    return ctrlBr.getTrueResult() == fstOperand;

  // The cmerge is consumed in an intermediate block. Find out if this block is
  // a predecessor of the "true" successor of the cmerge.
  Block *trueBlock = trueUser->getBlock();
  return domInfo.dominates(trueBlock, fstOperand.getDefiningOp()->getBlock());
}

namespace {
// This function creates the loop 'continue' and 'exit' network around backedges
// in the CFG.
// We don't have a standard dialect based LoopInfo utility in MLIR
// (which could probably deduce most of the information that we need for this
// transformation), so we roll our own loop-detection analysis. This is
// simplified by the need to only detect outermost loops. Inner loops are
// not included in the loop network (since we only care about restricting
// different function invocations from activating a loop, not prevent loop
// pipelining within a single function invocation).
class LoopNetworkRewriter {
public:
  LoopNetworkRewriter(HandshakeLowering &hl) : hl(hl) {}

  LogicalResult processRegion(Region &r, ConversionPatternRewriter &rewriter);

private:
  // An exit pair is a pair of <in loop block, outside loop block> that
  // indicates where control leaves a loop.
  using ExitPair = std::pair<Block *, Block *>;
  LogicalResult processOuterLoop(Location loc, CFGLoop *loop);

  // Builds the loop continue network in between the loop header and its loop
  // latch. The loop continuation network will replace the existing control
  // merge in the loop header with a mux + loop priming register.
  // The 'loopPrimingInput' is a backedge that will later be assigned by
  // 'buildExitNetwork'. The value is to be used as the input to the loop
  // priming buffer.
  // Returns a reference to the loop priming register.
  BufferOp buildContinueNetwork(Block *loopHeader, Block *loopLatch,
                                Backedge &loopPrimingInput);

  // Builds the loop exit network. This detects the conditional operands used in
  // each of the exit blocks, matches their parity with the convention used to
  // prime the loop register, and assigns it to the loop priming register input.
  void buildExitNetwork(Block *loopHeader,
                        const llvm::SmallSet<ExitPair, 2> &exitPairs,
                        BufferOp loopPrimingRegister,
                        Backedge &loopPrimingInput);

private:
  ConversionPatternRewriter *rewriter = nullptr;
  HandshakeLowering &hl;
};
} // namespace

LogicalResult
HandshakeLowering::loopNetworkRewriting(ConversionPatternRewriter &rewriter) {
  return LoopNetworkRewriter(*this).processRegion(r, rewriter);
}

LogicalResult
LoopNetworkRewriter::processRegion(Region &r,
                                   ConversionPatternRewriter &rewriter) {
  // Nothing to do on a single block region.
  if (r.hasOneBlock())
    return success();
  this->rewriter = &rewriter;

  Operation *op = r.getParentOp();

  DominanceInfo domInfo(op);
  CFGLoopInfo loopInfo(domInfo.getDomTree(&r));

  for (CFGLoop *loop : loopInfo.getTopLevelLoops()) {
    if (!loop->getLoopLatch())
      return emitError(op->getLoc()) << "Multiple loop latches detected "
                                        "(backedges from within the loop "
                                        "to the loop header). Loop task "
                                        "pipelining is only supported for "
                                        "loops with unified loop latches.";

    // This is the start of an outer loop - go process!
    if (failed(processOuterLoop(op->getLoc(), loop)))
      return failure();
  }

  return success();
}

// Returns the operand of the 'mux' operation which originated from 'block'.
static Value getOperandFromBlock(MuxOp mux, Block *block) {
  auto inValueIt = llvm::find_if(mux.getDataOperands(), [&](Value operand) {
    return block == operand.getParentBlock();
  });
  assert(
      inValueIt != mux.getOperands().end() &&
      "Expected mux to have an operand originating from the requested block.");
  return *inValueIt;
}

// Returns a list of operands from 'mux' which corresponds to the inputs of the
// 'cmerge' operation. The results are sorted such that the i'th cmerge operand
// and the i'th sorted operand originate from the same block.
static std::vector<Value> getSortedInputs(ControlMergeOp cmerge, MuxOp mux) {
  std::vector<Value> sortedOperands;
  for (auto in : cmerge.getOperands()) {
    auto *srcBlock = in.getParentBlock();
    sortedOperands.push_back(getOperandFromBlock(mux, srcBlock));
  }

  // Sanity check: ensure that operands are unique
  for (unsigned i = 0; i < sortedOperands.size(); ++i) {
    for (unsigned j = 0; j < sortedOperands.size(); ++j) {
      if (i == j)
        continue;
      assert(sortedOperands[i] != sortedOperands[j] &&
             "Cannot have an identical operand from two different blocks!");
    }
  }

  return sortedOperands;
}

BufferOp LoopNetworkRewriter::buildContinueNetwork(Block *loopHeader,
                                                   Block *loopLatch,
                                                   Backedge &loopPrimingInput) {
  // Gather the muxes to replace before modifying block internals; it's been
  // found that if this is not done, we have determinism issues wrt. generating
  // the same order of replaced muxes on repeated runs of an identical
  // conversion.
  llvm::SmallVector<MuxOp> muxesToReplace;
  llvm::copy(loopHeader->getOps<MuxOp>(), std::back_inserter(muxesToReplace));

  // Fetch the control merge of the block; it is assumed that, at this point of
  // lowering, no other form of control can be used for the loop header block
  // than a control merge.
  auto *cmerge = getControlMerge(loopHeader);
  assert(hl.getBlockEntryControl(loopHeader) == cmerge->getResult(0) &&
         "Expected control merge to be the control component of a loop header");
  auto loc = cmerge->getLoc();

  // sanity check: cmerge should have >1 input to actually be a loop
  assert(cmerge->getNumOperands() > 1 && "This cannot be a loop header");

  // Partition the control merge inputs into those originating from backedges,
  // and those originating elsewhere.
  SmallVector<Value> externalCtrls, loopCtrls;
  for (auto cval : cmerge->getOperands()) {
    if (cval.getParentBlock() == loopLatch)
      loopCtrls.push_back(cval);
    else
      externalCtrls.push_back(cval);
  }
  assert(loopCtrls.size() == 1 &&
         "Expected a single loop control value to match the single loop latch");
  Value loopCtrl = loopCtrls.front();

  // Merge all of the controls in each partition
  rewriter->setInsertionPointToStart(loopHeader);
  auto externalCtrlMerge = rewriter->create<ControlMergeOp>(loc, externalCtrls);

  // Create loop mux and the loop priming register. The loop mux will on select
  // "0" select external control, and internal control at "1". This convention
  // must be followed by the loop exit network.
  auto primingRegister =
      rewriter->create<BufferOp>(loc, loopPrimingInput, 1, BufferTypeEnum::seq);
  // Initialize the priming register to path 0.
  primingRegister->setAttr("initValues", rewriter->getI64ArrayAttr({0}));

  // The loop control mux will deterministically select between control entering
  // the loop from any external block or the single loop backedge.
  auto loopCtrlMux = rewriter->create<MuxOp>(
      loc, primingRegister.getResult(),
      llvm::SmallVector<Value>{externalCtrlMerge.getResult(), loopCtrl});

  // Replace the existing control merge 'result' output with the loop control
  // mux.
  cmerge->getResult(0).replaceAllUsesWith(loopCtrlMux.getResult());

  // Register the new block entry control value
  hl.setBlockEntryControl(loopHeader, loopCtrlMux.getResult());

  // Next, we need to consider how to replace the control merge 'index' output,
  // used to drive input selection to the block.

  // Inputs to the loop header will be sourced from muxes with inputs from both
  // the loop latch as well as external blocks. Iterate through these and sort
  // based on the input ordering to the external/internal control merge.
  // We do this by maintaining a mapping between the external and loop data
  // inputs for each data mux in the design. The key of these maps is the
  // original mux (that is to be replaced).
  DenseMap<MuxOp, std::vector<Value>> externalDataInputs;
  DenseMap<MuxOp, Value> loopDataInputs;
  for (auto muxOp : muxesToReplace) {
    if (muxOp == loopCtrlMux)
      continue;

    externalDataInputs[muxOp] = getSortedInputs(externalCtrlMerge, muxOp);
    loopDataInputs[muxOp] = getOperandFromBlock(muxOp, loopLatch);
    assert(/*loop latch input*/ 1 + externalDataInputs[muxOp].size() ==
               muxOp.getDataOperands().size() &&
           "Expected all mux operands to be partitioned between loop and "
           "external data inputs");
  }

  // With this, we now replace each of the data input muxes in the loop header.
  // We instantiate a single mux driven by the external control merge.
  // This, as well as the corresponding data input coming from within the single
  // loop latch, will then be selected between by a 3rd mux, based on the
  // priming register.
  for (MuxOp mux : muxesToReplace) {
    auto externalDataMux = rewriter->create<MuxOp>(
        loc, externalCtrlMerge.getIndex(), externalDataInputs[mux]);

    rewriter->replaceOp(
        mux, rewriter
                 ->create<MuxOp>(loc, primingRegister,
                                 llvm::SmallVector<Value>{externalDataMux,
                                                          loopDataInputs[mux]})
                 .getResult());
  }

  // Now all values defined by the original cmerge should have been replaced,
  // and it may be erased.
  rewriter->eraseOp(cmerge);

  // Return the priming register to be referenced by the exit network builder.
  return primingRegister;
}

void LoopNetworkRewriter::buildExitNetwork(
    Block *loopHeader, const llvm::SmallSet<ExitPair, 2> &exitPairs,
    BufferOp loopPrimingRegister, Backedge &loopPrimingInput) {
  auto loc = loopPrimingRegister.getLoc();

  // Iterate over the exit pairs to gather up the condition signals that need to
  // be connected to the exit network. In doing so, we parity-correct these
  // condition values based on the convention used in buildContinueNetwork - The
  // loop mux will on select "0" select external control, and internal control
  // at "1". This convention which must be followed by the loop exit network.
  // External control must be selected when exiting the loop (to reprime the
  // register).
  SmallVector<Value> parityCorrectedConds;
  for (auto &[condBlock, exitBlock] : exitPairs) {
    auto condBr = getControlCondBranch(condBlock);
    assert(
        condBr &&
        "Expected a conditional control branch op in the loop condition block");
    Operation *trueUser = *condBr.getTrueResult().getUsers().begin();
    bool isTrueParity = trueUser->getBlock() == exitBlock;
    assert(isTrueParity ^
               ((*condBr.getFalseResult().getUsers().begin())->getBlock() ==
                exitBlock) &&
           "The user of either the true or the false result should be in the "
           "exit block");

    Value condValue = condBr.getConditionOperand();
    if (isTrueParity) {
      // This goes against the convention, and we have to invert the condition
      // value before connecting it to the exit network.
      rewriter->setInsertionPoint(condBr);
      condValue = rewriter->create<arith::XOrIOp>(
          loc, condValue.getType(), condValue,
          rewriter->create<arith::ConstantOp>(
              loc, rewriter->getIntegerAttr(rewriter->getI1Type(), 1)));
    }
    parityCorrectedConds.push_back(condValue);
  }

  // Merge all of the parity-corrected exit conditions and assign them
  // to the loop priming input.
  auto exitMerge = rewriter->create<MergeOp>(loc, parityCorrectedConds);
  loopPrimingInput.setValue(exitMerge);
}

LogicalResult LoopNetworkRewriter::processOuterLoop(Location loc,
                                                    CFGLoop *loop) {
  // We determine the exit pairs of the loop; this is the in-loop nodes
  // which branch off to the exit nodes.
  llvm::SmallSet<ExitPair, 2> exitPairs;
  SmallVector<Block *> exitBlocks;
  loop->getExitBlocks(exitBlocks);
  for (auto *exitNode : exitBlocks) {
    for (auto *pred : exitNode->getPredecessors()) {
      // is the predecessor inside the loop?
      if (!loop->contains(pred))
        continue;

      ExitPair condPair = {pred, exitNode};
      assert(!exitPairs.count(condPair) &&
             "identical condition pairs should never be possible");
      exitPairs.insert(condPair);
    }
  }
  assert(!exitPairs.empty() && "No exits from loop?");

  // The first precondition to our loop transformation is that only a single
  // exit pair exists in the loop.
  if (exitPairs.size() > 1)
    return emitError(loc)
           << "Multiple exits detected within a loop. Loop task pipelining is "
              "only supported for loops with unified loop exit blocks.";

  Block *header = loop->getHeader();
  BackedgeBuilder bebuilder(*rewriter, header->front().getLoc());

  // Build the loop continue network. Loop continuation is triggered solely by
  // backedges to the header.
  auto loopPrimingRegisterInput = bebuilder.get(rewriter->getI1Type());
  auto loopPrimingRegister = buildContinueNetwork(header, loop->getLoopLatch(),
                                                  loopPrimingRegisterInput);

  // Build the loop exit network. Loop exiting is driven solely by exit pairs
  // from the loop.
  buildExitNetwork(header, exitPairs, loopPrimingRegister,
                   loopPrimingRegisterInput);

  return success();
}

// Return the appropriate branch result based on successor block which uses it
static Value getSuccResult(Operation *termOp, Operation *newOp,
                           Block *succBlock) {
  // For conditional block, check if result goes to true or to false successor
  if (auto condBranchOp = dyn_cast<mlir::cf::CondBranchOp>(termOp)) {
    if (condBranchOp.getTrueDest() == succBlock)
      return dyn_cast<handshake::ConditionalBranchOp>(newOp).getTrueResult();
    else {
      assert(condBranchOp.getFalseDest() == succBlock);
      return dyn_cast<handshake::ConditionalBranchOp>(newOp).getFalseResult();
    }
  }
  // If the block is unconditional, newOp has only one result
  return newOp->getResult(0);
}

LogicalResult
HandshakeLowering::addBranchOps(ConversionPatternRewriter &rewriter) {

  BlockValues liveOuts;

  for (Block &block : r) {
    for (Operation &op : block) {
      for (auto result : op.getResults())
        if (isLiveOut(result))
          liveOuts[&block].push_back(result);
    }
  }

  for (Block &block : r) {
    Operation *termOp = block.getTerminator();
    rewriter.setInsertionPoint(termOp);

    for (Value val : liveOuts[&block]) {
      // Count the number of branches which the liveout needs
      int numBranches = getBranchCount(val, &block);

      // Instantiate branches and connect to Merges
      for (int i = 0, e = numBranches; i < e; ++i) {
        Operation *newOp = nullptr;

        if (auto condBranchOp = dyn_cast<mlir::cf::CondBranchOp>(termOp))
          newOp = rewriter.create<handshake::ConditionalBranchOp>(
              termOp->getLoc(), condBranchOp.getCondition(), val);
        else if (isa<mlir::cf::BranchOp>(termOp))
          newOp = rewriter.create<handshake::BranchOp>(termOp->getLoc(), val);

        if (newOp == nullptr)
          continue;

        for (int j = 0, e = block.getNumSuccessors(); j < e; ++j) {
          Block *succ = block.getSuccessor(j);
          Value res = getSuccResult(termOp, newOp, succ);

          for (auto &u : val.getUses()) {
            if (u.getOwner()->getBlock() == succ) {
              u.getOwner()->replaceUsesOfWith(val, res);
              break;
            }
          }
        }
      }
    }
  }

  return success();
}

LogicalResult HandshakeLowering::connectConstantsToControl(
    ConversionPatternRewriter &rewriter, bool sourceConstants) {
  // Create new constants which have a control-only input to trigger them.
  // These are conneted to the control network or optionally to a Source
  // operation (always triggering). Control-network connected constants may
  // help debugability, but result in a slightly larger circuit.

  if (sourceConstants) {
    for (auto constantOp : llvm::make_early_inc_range(
             r.template getOps<mlir::arith::ConstantOp>())) {
      rewriter.setInsertionPointAfter(constantOp);
      auto value = constantOp.getValue();
      rewriter.replaceOpWithNewOp<handshake::ConstantOp>(
          constantOp, value.getType(), value,
          rewriter.create<handshake::SourceOp>(constantOp.getLoc(),
                                               rewriter.getNoneType()));
    }
  } else {
    for (Block &block : r) {
      Value blockEntryCtrl = getBlockEntryControl(&block);
      for (auto constantOp : llvm::make_early_inc_range(
               block.template getOps<mlir::arith::ConstantOp>())) {
        rewriter.setInsertionPointAfter(constantOp);
        auto value = constantOp.getValue();
        rewriter.replaceOpWithNewOp<handshake::ConstantOp>(
            constantOp, value.getType(), value, blockEntryCtrl);
      }
    }
  }
  return success();
}

/// Holds information about an handshake "basic block terminator" control
/// operation
struct BlockControlTerm {
  /// The operation
  Operation *op;
  /// The operation's control operand (must have type NoneType)
  Value ctrlOperand;

  BlockControlTerm(Operation *op, Value ctrlOperand)
      : op(op), ctrlOperand(ctrlOperand) {
    assert(op && ctrlOperand);
    assert(ctrlOperand.getType().isa<NoneType>() &&
           "Control operand must be a NoneType");
  }

  /// Checks for member-wise equality
  friend bool operator==(const BlockControlTerm &lhs,
                         const BlockControlTerm &rhs) {
    return lhs.op == rhs.op && lhs.ctrlOperand == rhs.ctrlOperand;
  }
};

static BlockControlTerm getBlockControlTerminator(Block *block) {
  // Identify the control terminator operation and its control operand in the
  // given block. One such operation must exist in the block
  for (Operation &op : *block) {
    if (auto branchOp = dyn_cast<handshake::BranchOp>(op))
      if (branchOp.isControl())
        return {branchOp, branchOp.getDataOperand()};
    if (auto branchOp = dyn_cast<handshake::ConditionalBranchOp>(op))
      if (branchOp.isControl())
        return {branchOp, branchOp.getDataOperand()};
    if (auto endOp = dyn_cast<handshake::ReturnOp>(op))
      return {endOp, endOp.getOperands().back()};
  }
  llvm_unreachable("Block terminator must exist");
}

static LogicalResult getOpMemRef(Operation *op, Value &out) {
  out = Value();
  if (auto memOp = dyn_cast<memref::LoadOp>(op))
    out = memOp.getMemRef();
  else if (auto memOp = dyn_cast<memref::StoreOp>(op))
    out = memOp.getMemRef();
  else if (isa<AffineReadOpInterface, AffineWriteOpInterface>(op)) {
    MemRefAccess access(op);
    out = access.memref;
  }
  if (out != Value())
    return success();
  return op->emitOpError("Unknown Op type");
}

static bool isMemoryOp(Operation *op) {
  return isa<memref::LoadOp, memref::StoreOp, AffineReadOpInterface,
             AffineWriteOpInterface>(op);
}

LogicalResult
HandshakeLowering::replaceMemoryOps(ConversionPatternRewriter &rewriter,
                                    MemRefToMemoryAccessOp &memRefOps) {

  std::vector<Operation *> opsToErase;

  // Enrich the memRefOps context with BlockArguments, in case they aren't used.
  for (auto arg : r.getArguments()) {
    auto memrefType = dyn_cast<mlir::MemRefType>(arg.getType());
    if (!memrefType)
      continue;
    // Ensure that this is a valid memref-typed value.
    if (failed(isValidMemrefType(arg.getLoc(), memrefType)))
      return failure();
    memRefOps.insert(std::make_pair(arg, std::vector<Operation *>()));
  }

  // Replace load and store ops with the corresponding handshake ops
  // Need to traverse ops in blocks to store them in memRefOps in program
  // order
  for (Operation &op : r.getOps()) {
    if (!isMemoryOp(&op))
      continue;

    rewriter.setInsertionPoint(&op);
    Value memref;
    if (getOpMemRef(&op, memref).failed())
      return failure();
    Operation *newOp = nullptr;

    llvm::TypeSwitch<Operation *>(&op)
        .Case<memref::LoadOp>([&](auto loadOp) {
          // Get operands which correspond to address indices
          // This will add all operands except alloc
          SmallVector<Value, 8> operands(loadOp.getIndices());

          newOp =
              rewriter.create<handshake::LoadOp>(op.getLoc(), memref, operands);
          op.getResult(0).replaceAllUsesWith(newOp->getResult(0));
        })
        .Case<memref::StoreOp>([&](auto storeOp) {
          // Get operands which correspond to address indices
          // This will add all operands except alloc and data
          SmallVector<Value, 8> operands(storeOp.getIndices());

          // Create new op where operands are store data and address indices
          newOp = rewriter.create<handshake::StoreOp>(
              op.getLoc(), storeOp.getValueToStore(), operands);
        })
        .Case<AffineReadOpInterface, AffineWriteOpInterface>([&](auto) {
          // Get essential memref access inforamtion.
          MemRefAccess access(&op);
          // The address of an affine load/store operation can be a result
          // of an affine map, which is a linear combination of constants
          // and parameters. Therefore, we should extract the affine map of
          // each address and expand it into proper expressions that
          // calculate the result.
          AffineMap map;
          if (auto loadOp = dyn_cast<AffineReadOpInterface>(op))
            map = loadOp.getAffineMap();
          else
            map = dyn_cast<AffineWriteOpInterface>(op).getAffineMap();

          // The returned object from expandAffineMap is an optional list of
          // the expansion results from the given affine map, which are the
          // actual address indices that can be used as operands for
          // handshake LoadOp/StoreOp. The following processing requires it
          // to be a valid result.
          auto operands =
              expandAffineMap(rewriter, op.getLoc(), map, access.indices);
          assert(operands && "Address operands of affine memref access "
                             "cannot be reduced.");

          if (isa<AffineReadOpInterface>(op)) {
            auto loadOp = rewriter.create<handshake::LoadOp>(
                op.getLoc(), access.memref, *operands);
            newOp = loadOp;
            op.getResult(0).replaceAllUsesWith(loadOp.getDataResult());
          } else {
            newOp = rewriter.create<handshake::StoreOp>(
                op.getLoc(), op.getOperand(0), *operands);
          }
        })
        .Default([&](auto) {
          op.emitOpError("Load/store operation cannot be handled.");
        });

    memRefOps[memref].push_back(newOp);
    opsToErase.push_back(&op);
  }

  // Erase old memory ops
  for (unsigned i = 0, e = opsToErase.size(); i != e; ++i) {
    auto *op = opsToErase[i];
    for (int j = 0, e = op->getNumOperands(); j < e; ++j)
      op->eraseOperand(0);
    assert(op->getNumOperands() == 0);

    rewriter.eraseOp(op);
  }

  return success();
}

static SmallVector<Value, 8> getResultsToMemory(Operation *op) {
  // Get load/store results which are given as inputs to MemoryOp

  if (handshake::LoadOp loadOp = dyn_cast<handshake::LoadOp>(op)) {
    // For load, get all address outputs/indices
    // (load also has one data output which goes to successor operation)
    SmallVector<Value, 8> results(loadOp.getAddressResults());
    return results;

  } else {
    // For store, all outputs (data and address indices) go to memory
    assert(dyn_cast<handshake::StoreOp>(op));
    handshake::StoreOp storeOp = dyn_cast<handshake::StoreOp>(op);
    SmallVector<Value, 8> results(storeOp.getResults());
    return results;
  }
}

static void addLazyForks(Region &f, ConversionPatternRewriter &rewriter) {

  for (Block &block : f) {
    Value ctrl = getBlockControlTerminator(&block).ctrlOperand;
    if (!ctrl.hasOneUse())
      insertFork(ctrl, true, rewriter);
  }
}

static void removeUnusedAllocOps(Region &r,
                                 ConversionPatternRewriter &rewriter) {
  std::vector<Operation *> opsToDelete;

  // Remove alloc operations whose result have no use
  for (auto &op : r.getOps())
    if (isAllocOp(&op) && op.getResult(0).use_empty())
      opsToDelete.push_back(&op);

  llvm::for_each(opsToDelete, [&](auto allocOp) { rewriter.eraseOp(allocOp); });
}

static void addJoinOps(ConversionPatternRewriter &rewriter,
                       ArrayRef<BlockControlTerm> controlTerms) {
  for (auto term : controlTerms) {
    auto &[op, ctrl] = term;
    auto *srcOp = ctrl.getDefiningOp();

    // Insert only single join per block
    if (!isa<JoinOp>(srcOp)) {
      rewriter.setInsertionPointAfter(srcOp);
      Operation *newJoin = rewriter.create<JoinOp>(srcOp->getLoc(), ctrl);
      op->replaceUsesOfWith(ctrl, newJoin->getResult(0));
    }
  }
}

static std::vector<BlockControlTerm>
getControlTerminators(ArrayRef<Operation *> memOps) {
  std::vector<BlockControlTerm> terminators;

  for (Operation *op : memOps) {
    // Get block from which the mem op originates
    Block *block = op->getBlock();
    // Identify the control terminator in the block
    auto term = getBlockControlTerminator(block);
    if (std::find(terminators.begin(), terminators.end(), term) ==
        terminators.end())
      terminators.push_back(term);
  }
  return terminators;
}

static void addValueToOperands(Operation *op, Value val) {

  SmallVector<Value, 8> results(op->getOperands());
  results.push_back(val);
  op->setOperands(results);
}

static void setLoadDataInputs(ArrayRef<Operation *> memOps, Operation *memOp) {
  // Set memory outputs as load input data
  int ld_count = 0;
  for (auto *op : memOps) {
    if (isa<handshake::LoadOp>(op))
      addValueToOperands(op, memOp->getResult(ld_count++));
  }
}

static LogicalResult setJoinControlInputs(ArrayRef<Operation *> memOps,
                                          Operation *memOp, int offset,
                                          ArrayRef<int> cntrlInd) {
  // Connect all memory ops to the join of that block (ensures that all mem
  // ops terminate before a new block starts)
  for (int i = 0, e = memOps.size(); i < e; ++i) {
    auto *op = memOps[i];
    Value ctrl = getBlockControlTerminator(op->getBlock()).ctrlOperand;
    auto *srcOp = ctrl.getDefiningOp();
    if (!isa<JoinOp>(srcOp)) {
      return srcOp->emitOpError("Op expected to be a JoinOp");
    }
    addValueToOperands(srcOp, memOp->getResult(offset + cntrlInd[i]));
  }
  return success();
}

void HandshakeLowering::setMemOpControlInputs(
    ConversionPatternRewriter &rewriter, ArrayRef<Operation *> memOps,
    Operation *memOp, int offset, ArrayRef<int> cntrlInd) {
  for (int i = 0, e = memOps.size(); i < e; ++i) {
    std::vector<Value> controlOperands;
    Operation *currOp = memOps[i];
    Block *currBlock = currOp->getBlock();

    // Set load/store control inputs from the block input control value
    Value blockEntryCtrl = getBlockEntryControl(currBlock);
    controlOperands.push_back(blockEntryCtrl);

    // Set load/store control inputs from predecessors in block
    for (int j = 0, f = i; j < f; ++j) {
      Operation *predOp = memOps[j];
      Block *predBlock = predOp->getBlock();
      if (currBlock == predBlock)
        // Any dependency but RARs
        if (!(isa<handshake::LoadOp>(currOp) && isa<handshake::LoadOp>(predOp)))
          // cntrlInd maps memOps index to correct control output index
          controlOperands.push_back(memOp->getResult(offset + cntrlInd[j]));
    }

    // If there is only one control input, add directly to memory op
    if (controlOperands.size() == 1)
      addValueToOperands(currOp, controlOperands[0]);

    // If multiple, join them and connect join output to memory op
    else {
      rewriter.setInsertionPoint(currOp);
      Operation *joinOp =
          rewriter.create<JoinOp>(currOp->getLoc(), controlOperands);
      addValueToOperands(currOp, joinOp->getResult(0));
    }
  }
}

LogicalResult
HandshakeLowering::connectToMemory(ConversionPatternRewriter &rewriter,
                                   MemRefToMemoryAccessOp memRefOps, bool lsq) {
  // Add MemoryOps which represent the memory interface
  // Connect memory operations and control appropriately
  int mem_count = 0;
  for (auto memory : memRefOps) {
    // First operand corresponds to memref (alloca or function argument)
    Value memrefOperand = memory.first;

    // A memory is external if the memref that defines it is provided as a
    // function (block) argument.
    bool isExternalMemory = memrefOperand.isa<BlockArgument>();

    mlir::MemRefType memrefType =
        memrefOperand.getType().cast<mlir::MemRefType>();
    if (failed(isValidMemrefType(memrefOperand.getLoc(), memrefType)))
      return failure();

    std::vector<Value> operands;

    // Get control terminators whose control operand need to connect to memory
    std::vector<BlockControlTerm> controlTerms =
        getControlTerminators(memory.second);

    // In case of LSQ interface, set control values as inputs (used to
    // trigger allocation to LSQ)
    if (lsq)
      for (auto valOp : controlTerms)
        operands.push_back(valOp.ctrlOperand);

    // Add load indices and store data+indices to memory operands
    // Count number of loads so that we can generate appropriate number of
    // memory outputs (data to load ops)

    // memory.second is in program order
    // Enforce MemoryOp port ordering as follows:
    // Operands: all stores then all loads (stdata1, staddr1, stdata2,...,
    // ldaddr1, ldaddr2,....) Outputs: all load outputs, ordered the same as
    // load addresses (lddata1, lddata2, ...), followed by all none outputs,
    // ordered as operands (stnone1, stnone2,...ldnone1, ldnone2,...)
    std::vector<int> newInd(memory.second.size(), 0);
    int ind = 0;
    for (int i = 0, e = memory.second.size(); i < e; ++i) {
      auto *op = memory.second[i];
      if (isa<handshake::StoreOp>(op)) {
        SmallVector<Value, 8> results = getResultsToMemory(op);
        operands.insert(operands.end(), results.begin(), results.end());
        newInd[i] = ind++;
      }
    }

    int ld_count = 0;

    for (int i = 0, e = memory.second.size(); i < e; ++i) {
      auto *op = memory.second[i];
      if (isa<handshake::LoadOp>(op)) {
        SmallVector<Value, 8> results = getResultsToMemory(op);
        operands.insert(operands.end(), results.begin(), results.end());

        ld_count++;
        newInd[i] = ind++;
      }
    }

    // control-only outputs for each access (indicate access completion)
    int cntrl_count = lsq ? 0 : memory.second.size();

    Block *entryBlock = &r.front();
    rewriter.setInsertionPointToStart(entryBlock);

    // Place memory op next to the alloc op
    Operation *newOp = nullptr;
    if (isExternalMemory)
      newOp = rewriter.create<ExternalMemoryOp>(
          entryBlock->front().getLoc(), memrefOperand, operands, ld_count,
          cntrl_count - ld_count, mem_count++);
    else
      newOp = rewriter.create<MemoryOp>(entryBlock->front().getLoc(), operands,
                                        ld_count, cntrl_count, lsq, mem_count++,
                                        memrefOperand);

    setLoadDataInputs(memory.second, newOp);

    if (!lsq) {
      // Create Joins which join done signals from memory with the
      // control-only network
      addJoinOps(rewriter, controlTerms);

      // Connect all load/store done signals to the join of their block
      // Ensure that the block terminates only after all its accesses have
      // completed
      // True is default. When no sync needed, set to false (for now,
      // user-determined)
      bool control = true;

      if (control)
        returnOnError(
            setJoinControlInputs(memory.second, newOp, ld_count, newInd));

      // Set control-only inputs to each memory op
      // Ensure that op starts only after prior blocks have completed
      // Ensure that op starts only after predecessor ops (with RAW, WAR, or
      // WAW) have completed
      setMemOpControlInputs(rewriter, memory.second, newOp, ld_count, newInd);
    }
  }

  if (lsq)
    addLazyForks(r, rewriter);

  removeUnusedAllocOps(r, rewriter);
  return success();
}

LogicalResult
HandshakeLowering::replaceCallOps(ConversionPatternRewriter &rewriter) {
  for (Block &block : r) {
    /// An instance is activated whenever control arrives at the basic block
    /// of the source callOp.
    Value blockEntryControl = getBlockEntryControl(&block);
    for (Operation &op : block) {
      if (auto callOp = dyn_cast<mlir::func::CallOp>(op)) {
        llvm::SmallVector<Value> operands;
        llvm::copy(callOp.getOperands(), std::back_inserter(operands));
        operands.push_back(blockEntryControl);
        rewriter.setInsertionPoint(callOp);
        auto instanceOp = rewriter.create<handshake::InstanceOp>(
            callOp.getLoc(), callOp.getCallee(), callOp.getResultTypes(),
            operands);
        // Replace all results of the source callOp.
        for (auto it : llvm::zip(callOp.getResults(), instanceOp.getResults()))
          std::get<0>(it).replaceAllUsesWith(std::get<1>(it));
        rewriter.eraseOp(callOp);
      }
    }
  }
  return success();
}

namespace {
/// Strategy class for SSA maximization during std-to-handshake conversion.
/// Block arguments of type MemRefType and allocation operations are not
/// considered for SSA maximization.
class HandshakeLoweringSSAStrategy : public SSAMaximizationStrategy {
  /// Filters out block arguments of type MemRefType
  bool maximizeArgument(BlockArgument arg) override {
    return !arg.getType().isa<mlir::MemRefType>();
  }

  /// Filters out allocation operations
  bool maximizeOp(Operation *op) override { return !isAllocOp(op); }
};
} // namespace

/// Converts every value in the region into maximal SSA form, unless the value
/// is a block argument of type MemRefType or the result of an allocation
/// operation.
static LogicalResult maximizeSSANoMem(Region &r,
                                      ConversionPatternRewriter &rewriter) {
  HandshakeLoweringSSAStrategy strategy;
  return maximizeSSA(r, strategy, rewriter);
}

static LogicalResult lowerFuncOp(func::FuncOp funcOp, MLIRContext *ctx,
                                 bool sourceConstants,
                                 bool disableTaskPipelining) {
  // Only retain those attributes that are not constructed by build.
  SmallVector<NamedAttribute, 4> attributes;
  for (const auto &attr : funcOp->getAttrs()) {
    if (attr.getName() == SymbolTable::getSymbolAttrName() ||
        attr.getName() == funcOp.getFunctionTypeAttrName())
      continue;
    attributes.push_back(attr);
  }

  // Get function arguments
  llvm::SmallVector<mlir::Type, 8> argTypes;
  for (auto &argType : funcOp.getArgumentTypes())
    argTypes.push_back(argType);

  // Get function results
  llvm::SmallVector<mlir::Type, 8> resTypes;
  for (auto resType : funcOp.getResultTypes())
    resTypes.push_back(resType);

  handshake::FuncOp newFuncOp;

  // Add control input/output to function arguments/results and create a
  // handshake::FuncOp of appropriate type
  returnOnError(partiallyLowerOp<func::FuncOp>(
      [&](func::FuncOp funcOp, PatternRewriter &rewriter) {
        auto noneType = rewriter.getNoneType();
        resTypes.push_back(noneType);
        argTypes.push_back(noneType);
        auto func_type = rewriter.getFunctionType(argTypes, resTypes);
        newFuncOp = rewriter.create<handshake::FuncOp>(
            funcOp.getLoc(), funcOp.getName(), func_type, attributes);
        rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                    newFuncOp.end());
        if (!newFuncOp.isExternal())
          newFuncOp.resolveArgAndResNames();
        rewriter.eraseOp(funcOp);
        return success();
      },
      ctx, funcOp));

  // Apply SSA maximization
  returnOnError(
      partiallyLowerRegion(maximizeSSANoMem, ctx, newFuncOp.getBody()));

  if (!newFuncOp.isExternal()) {
    HandshakeLowering fol(newFuncOp.getBody());
    returnOnError(lowerRegion<func::ReturnOp>(fol, sourceConstants,
                                              disableTaskPipelining));
  }

  return success();
}

namespace {

struct HandshakeRemoveBlockPass
    : HandshakeRemoveBlockBase<HandshakeRemoveBlockPass> {
  void runOnOperation() override { removeBasicBlocks(getOperation()); }
};

struct StandardToHandshakePass
    : public StandardToHandshakeBase<StandardToHandshakePass> {
  StandardToHandshakePass(bool sourceConstants, bool disableTaskPipelining) {
    this->sourceConstants = sourceConstants;
    this->disableTaskPipelining = disableTaskPipelining;
  }
  void runOnOperation() override {
    ModuleOp m = getOperation();

    for (auto funcOp : llvm::make_early_inc_range(m.getOps<func::FuncOp>())) {
      if (failed(lowerFuncOp(funcOp, &getContext(), sourceConstants,
                             disableTaskPipelining))) {
        signalPassFailure();
        return;
      }
    }

    // Legalize the resulting regions, removing basic blocks and performing
    // any simple conversions.
    for (auto func : m.getOps<handshake::FuncOp>())
      removeBasicBlocks(func);
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
circt::createStandardToHandshakePass(bool sourceConstants,
                                     bool disableTaskPipelining) {
  return std::make_unique<StandardToHandshakePass>(sourceConstants,
                                                   disableTaskPipelining);
}

std::unique_ptr<mlir::OperationPass<handshake::FuncOp>>
circt::createHandshakeRemoveBlockPass() {
  return std::make_unique<HandshakeRemoveBlockPass>();
}
