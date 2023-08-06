//===- Buffers.cpp - buffer materialization passes --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of buffer materialization passes.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace circt;
using namespace handshake;
using namespace mlir;

namespace {

struct RemoveHandshakeBuffers : public OpRewritePattern<handshake::BufferOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::BufferOp bufferOp,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOp(bufferOp, bufferOp.getOperand());
    return success();
  }
};

struct HandshakeRemoveBuffersPass
    : public HandshakeRemoveBuffersBase<HandshakeRemoveBuffersPass> {
  void runOnOperation() override {
    handshake::FuncOp op = getOperation();
    ConversionTarget target(getContext());
    target.addIllegalOp<handshake::BufferOp>();
    RewritePatternSet patterns(&getContext());
    patterns.insert<RemoveHandshakeBuffers>(&getContext());

    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      signalPassFailure();
  };
};
} // namespace
// Returns true if a block argument should have buffers added to its uses.
static bool shouldBufferArgument(BlockArgument arg) {
  // At the moment, buffers only make sense on arguments which we know
  // will lower down to a handshake bundle.
  return arg.getType().isIntOrFloat() || arg.getType().isa<NoneType>();
}

static bool isUnbufferedChannel(Operation *definingOp, Operation *usingOp) {
  return !isa_and_nonnull<BufferOp>(definingOp) && !isa<BufferOp>(usingOp);
}

static void insertBuffer(Location loc, Value operand, OpBuilder &builder,
                         unsigned numSlots, BufferTypeEnum bufferType) {
  auto ip = builder.saveInsertionPoint();
  builder.setInsertionPointAfterValue(operand);
  auto bufferOp =
      builder.create<handshake::BufferOp>(loc, operand, numSlots, bufferType);
  operand.replaceUsesWithIf(
      bufferOp, function_ref<bool(OpOperand &)>([](OpOperand &operand) -> bool {
        return !isa<handshake::BufferOp>(operand.getOwner());
      }));
  builder.restoreInsertionPoint(ip);
}

// Inserts buffers at all results of an operation
static void bufferResults(OpBuilder &builder, Operation *op, unsigned numSlots,
                          BufferTypeEnum bufferType) {
  for (auto res : op->getResults()) {
    Operation *user = *res.getUsers().begin();
    if (isa<handshake::BufferOp>(user))
      continue;
    insertBuffer(op->getLoc(), res, builder, numSlots, bufferType);
  }
}

// Add a buffer to any un-buffered channel.
static void bufferAllStrategy(Region &r, OpBuilder &builder, unsigned numSlots,
                              BufferTypeEnum bufferType = BufferTypeEnum::seq) {

  for (auto &arg : r.getArguments()) {
    if (!shouldBufferArgument(arg))
      continue;
    insertBuffer(arg.getLoc(), arg, builder, numSlots, bufferType);
  }

  for (auto &defOp : r.getOps()) {
    for (auto res : defOp.getResults()) {
      for (auto *useOp : res.getUsers()) {
        if (!isUnbufferedChannel(&defOp, useOp))
          continue;
        insertBuffer(res.getLoc(), res, builder, numSlots, bufferType);
      }
    }
  }
}

// Returns true if 'src' is within a cycle. 'breaksCycle' is a function which
// determines whether an operation breaks a cycle.
static bool inCycle(Operation *src,
                    llvm::function_ref<bool(Operation *)> breaksCycle) {
  SetVector<Operation *> visited;
  SmallVector<Operation *> stack = {src};

  while (!stack.empty()) {
    Operation *curr = stack.pop_back_val();

    if (visited.contains(curr))
      continue;
    visited.insert(curr);

    if (breaksCycle(curr))
      continue;

    for (auto *user : curr->getUsers()) {
      // If visiting the source node, then we're in a cycle.
      if (src == user)
        return true;

      stack.push_back(user);
    }
  }
  return false;
}

// Perform a depth first search and insert buffers when cycles are detected.
static void
bufferCyclesStrategy(Region &r, OpBuilder &builder, unsigned numSlots,
                     BufferTypeEnum /*bufferType*/ = BufferTypeEnum::seq) {
  // Cycles can only occur at merge-like operations so those are our buffering
  // targets. Placing the buffer at the output of the merge-like op,
  // as opposed to naivly placing buffers *whenever* cycles are detected
  // ensures that we don't place a bunch of buffers on each input of the
  // merge-like op.
  auto isSeqBuffer = [](auto op) {
    auto bufferOp = dyn_cast<handshake::BufferOp>(op);
    return bufferOp && bufferOp.isSequential();
  };

  for (auto mergeOp : r.getOps<MergeLikeOpInterface>()) {
    // We insert a sequential buffer whenever the op is determined to be
    // within a cycle (to break combinational cycles). Else, place a FIFO
    // buffer.
    bool sequential = inCycle(mergeOp, isSeqBuffer);
    bufferResults(builder, mergeOp, numSlots,
                  sequential ? BufferTypeEnum::seq : BufferTypeEnum::fifo);
  }
}

// Combination of bufferCyclesStrategy and bufferAllStrategy, where we add a
// sequential buffer on graph cycles, and add FIFO buffers on all other
// connections.
static void bufferAllFIFOStrategy(Region &r, OpBuilder &builder,
                                  unsigned numSlots) {
  // First, buffer cycles with sequential buffers
  bufferCyclesStrategy(r, builder, /*numSlots=*/numSlots,
                       /*bufferType=*/BufferTypeEnum::seq);
  // Then, buffer remaining channels with transparent FIFO buffers
  bufferAllStrategy(r, builder, numSlots,
                    /*bufferType=*/BufferTypeEnum::fifo);
}

LogicalResult circt::handshake::bufferRegion(Region &r, OpBuilder &builder,
                                             StringRef strategy,
                                             unsigned bufferSize) {
  if (strategy == "cycles")
    bufferCyclesStrategy(r, builder, bufferSize);
  else if (strategy == "all")
    bufferAllStrategy(r, builder, bufferSize);
  else if (strategy == "allFIFO")
    bufferAllFIFOStrategy(r, builder, bufferSize);
  else
    return r.getParentOp()->emitOpError()
           << "Unknown buffer strategy: " << strategy;

  return success();
}

namespace {
struct HandshakeInsertBuffersPass
    : public HandshakeInsertBuffersBase<HandshakeInsertBuffersPass> {
  HandshakeInsertBuffersPass(const std::string &strategy, unsigned bufferSize) {
    this->strategy = strategy;
    this->bufferSize = bufferSize;
  }

  void runOnOperation() override {
    auto f = getOperation();
    if (f.isExternal())
      return;

    OpBuilder builder(f.getContext());

    if (failed(bufferRegion(f.getBody(), builder, strategy, bufferSize)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
circt::handshake::createHandshakeRemoveBuffersPass() {
  return std::make_unique<HandshakeRemoveBuffersPass>();
}

std::unique_ptr<mlir::OperationPass<handshake::FuncOp>>
circt::handshake::createHandshakeInsertBuffersPass(const std::string &strategy,
                                                   unsigned bufferSize) {
  return std::make_unique<HandshakeInsertBuffersPass>(strategy, bufferSize);
}
