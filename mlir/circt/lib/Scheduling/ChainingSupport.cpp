//===- ChainingSupport.cpp - Utilities for chaining-aware schedulers ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helpers to add chaining support to existing algorithms via chain-breaking
// auxiliary dependences, i.e. edges whose endpoints must be separated by at
// least one time steps.
//
//===----------------------------------------------------------------------===//

#include "circt/Scheduling/Utilities.h"

#include "mlir/IR/Operation.h"

using namespace circt;
using namespace circt::scheduling;

using Dependence = Problem::Dependence;

LogicalResult scheduling::computeChainBreakingDependences(
    ChainingProblem &prob, float cycleTime,
    SmallVectorImpl<Dependence> &result) {
  // Sanity check: This approach treats the given `cycleTime` as a hard
  // constraint, so all individual delays must be shorter.
  for (auto opr : prob.getOperatorTypes())
    if (*prob.getIncomingDelay(opr) > cycleTime ||
        *prob.getOutgoingDelay(opr) > cycleTime)
      return prob.getContainingOp()->emitError()
             << "Delays of operator type '" << opr.getValue()
             << "' exceed maximum cycle time: " << cycleTime;

  // chains[v][u] denotes the accumulated delay incoming at `v`, of the longest
  // combinational chain originating from `u`.
  DenseMap<Operation *, SmallDenseMap<Operation *, float>> chains;

  // Do a simple DFA-style pass over the dependence graph to determine
  // combinational chains and their respective accumulated delays.
  return handleOperationsInTopologicalOrder(prob, [&](Operation *op) {
    // Mark `op` to be the origin of its own chain.
    chains[op][op] = 0.0f;

    for (auto dep : prob.getDependences(op)) {
      // Skip auxiliary deps, as these don't carry values.
      if (dep.isAuxiliary())
        continue;

      Operation *pred = dep.getSource();
      if (!chains.count(pred))
        return failure(); // Predecessor hasn't been handled yet.

      auto predOpr = *prob.getLinkedOperatorType(pred);
      if (*prob.getLatency(predOpr) > 0) {
        // `pred` is not combinational, so none of its incoming chains are
        // extended. Hence, it only contributes its outgoing delay to `op`'s
        // incoming delay.
        chains[op][pred] = *prob.getOutgoingDelay(predOpr);
        continue;
      }

      // Otherwise, `pred` is combinational. This means that all of its incoming
      // chains, extended by `pred`, are incoming chains for `op`.
      for (auto incomingChain : chains[pred]) {
        Operation *origin = incomingChain.first;
        float delay = incomingChain.second;
        chains[op][origin] = std::max(delay + *prob.getOutgoingDelay(predOpr),
                                      chains[op][origin]);
      }
    }

    // All chains/accumulated delays incoming at `op` are now known.
    auto opr = *prob.getLinkedOperatorType(op);
    for (auto incomingChain : chains[op]) {
      Operation *origin = incomingChain.first;
      float delay = incomingChain.second;
      // Check whether `op` could be appended to the incoming chain without
      // violating the cycle time constraint.
      if (delay + *prob.getIncomingDelay(opr) > cycleTime) {
        // If not, add a chain-breaking auxiliary dep ...
        result.emplace_back(origin, op);
        // ... and end the chain here.
        chains[op].erase(origin);
      }
    }

    return success();
  });
}

LogicalResult scheduling::computeStartTimesInCycle(ChainingProblem &prob) {
  return handleOperationsInTopologicalOrder(prob, [&](Operation *op) {
    // `op` will start within its abstract time step as soon as all operand
    // values have reached it.
    unsigned startTime = *prob.getStartTime(op);
    float startTimeInCycle = 0.0f;

    for (auto dep : prob.getDependences(op)) {
      // Skip auxiliary deps, as these don't carry values.
      if (dep.isAuxiliary())
        continue;

      Operation *pred = dep.getSource();
      auto predStartTimeInCycle = prob.getStartTimeInCycle(pred);
      if (!predStartTimeInCycle)
        return failure(); // Predecessor hasn't been handled yet.

      auto predOpr = *prob.getLinkedOperatorType(pred);
      unsigned predStartTime = *prob.getStartTime(pred);
      unsigned predEndTime = predStartTime + *prob.getLatency(predOpr);

      if (predEndTime < startTime)
        // Incoming value is completely registered/available with the beginning
        // of the cycle.
        continue;

      // So, `pred` ends in the same cycle as `op` starts.
      assert(predEndTime == startTime);
      // If `pred` uses a multi-cycle operator, only its outgoing delay counts.
      float predEndTimeInCycle =
          (predStartTime == predEndTime ? *predStartTimeInCycle : 0.0f) +
          *prob.getOutgoingDelay(predOpr);
      startTimeInCycle = std::max(predEndTimeInCycle, startTimeInCycle);
    }

    prob.setStartTimeInCycle(op, startTimeInCycle);
    return success();
  });
}
