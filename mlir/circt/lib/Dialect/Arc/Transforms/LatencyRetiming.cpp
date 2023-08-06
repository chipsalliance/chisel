//===- LatencyRetiming.cpp - Implement LatencyRetiming Pass ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/SymCache.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "arc-latency-retiming"

using namespace circt;
using namespace arc;

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

namespace {
struct LatencyRetimingStatistics {
  unsigned numOpsRemoved = 0;
  unsigned latencyUnitsSaved = 0;
};

/// Absorb the latencies from predecessor states to collapse shift registers and
/// reduce the overall amount of latency units in the design.
struct LatencyRetimingPattern : OpRewritePattern<StateOp> {
  LatencyRetimingPattern(MLIRContext *context, SymbolCache &symCache,
                         LatencyRetimingStatistics &statistics)
      : OpRewritePattern<StateOp>(context), symCache(symCache),
        statistics(statistics) {}

  LogicalResult matchAndRewrite(StateOp op,
                                PatternRewriter &rewriter) const final;

private:
  SymbolCache &symCache;
  LatencyRetimingStatistics &statistics;
};

} // namespace

LogicalResult
LatencyRetimingPattern::matchAndRewrite(StateOp op,
                                        PatternRewriter &rewriter) const {
  unsigned minPrevLatency = UINT_MAX;
  SetVector<StateOp> predecessors;

  if (op.getReset() || op.getEnable())
    return failure();

  for (auto input : op.getInputs()) {
    auto predState = input.getDefiningOp<StateOp>();
    if (!predState)
      return failure();

    if (predState->hasAttr("name") || predState->hasAttr("names"))
      return failure();

    if (predState == op)
      return failure();

    if (predState.getLatency() != 0 && op.getLatency() != 0 &&
        predState.getClock() != op.getClock())
      return failure();

    if (predState.getEnable() || predState.getReset())
      return failure();

    if (llvm::any_of(predState->getUsers(),
                     [&](auto *user) { return user != op; }))
      return failure();

    predecessors.insert(predState);
    minPrevLatency = std::min(minPrevLatency, predState.getLatency());
  }

  if (minPrevLatency == 0 || minPrevLatency == UINT_MAX)
    return failure();

  op.setLatency(op.getLatency() + minPrevLatency);
  for (auto prevStateOp : predecessors) {
    if (!op.getClock() && !op->getParentOfType<ClockDomainOp>())
      op.getClockMutable().assign(prevStateOp.getClock());

    statistics.latencyUnitsSaved += minPrevLatency;
    auto newLatency = prevStateOp.getLatency() - minPrevLatency;
    prevStateOp.setLatency(newLatency);

    if (newLatency > 0)
      continue;

    prevStateOp.getClockMutable().clear();
    if (cast<DefineOp>(symCache.getDefinition(prevStateOp.getArcAttr()))
            .isPassthrough()) {
      rewriter.replaceOp(prevStateOp, prevStateOp.getInputs());
      ++statistics.numOpsRemoved;
    }
  }
  statistics.latencyUnitsSaved -= minPrevLatency;

  return success();
}

//===----------------------------------------------------------------------===//
// LatencyRetiming pass
//===----------------------------------------------------------------------===//

namespace {
struct LatencyRetimingPass : LatencyRetimingBase<LatencyRetimingPass> {
  void runOnOperation() override;
};
} // namespace

void LatencyRetimingPass::runOnOperation() {
  SymbolCache cache;
  cache.addDefinitions(getOperation());

  LatencyRetimingStatistics statistics;

  RewritePatternSet patterns(&getContext());
  patterns.add<LatencyRetimingPattern>(&getContext(), cache, statistics);

  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();

  numOpsRemoved = statistics.numOpsRemoved;
  latencyUnitsSaved = statistics.latencyUnitsSaved;
}

std::unique_ptr<Pass> arc::createLatencyRetimingPass() {
  return std::make_unique<LatencyRetimingPass>();
}
