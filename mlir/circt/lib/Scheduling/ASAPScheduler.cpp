//===- ASAPScheduler.cpp - As-soon-as-possible list scheduler -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of an as-soon-as-possible list scheduler for acyclic problems.
//
//===----------------------------------------------------------------------===//

#include "circt/Scheduling/Algorithms.h"
#include "circt/Scheduling/Utilities.h"

using namespace circt;
using namespace circt::scheduling;

LogicalResult scheduling::scheduleASAP(Problem &prob) {
  return handleOperationsInTopologicalOrder(prob, [&](Operation *op) {
    // Operations with no predecessors are scheduled at time step 0
    if (prob.getDependences(op).empty()) {
      prob.setStartTime(op, 0);
      return success();
    }

    // op has at least one predecessor. Compute start time as:
    //   max_{p : preds} startTime[p] + latency[linkedOpr[p]]
    unsigned startTime = 0;
    for (auto &dep : prob.getDependences(op)) {
      Operation *pred = dep.getSource();
      auto predStart = prob.getStartTime(pred);
      if (!predStart)
        // pred is not yet scheduled, give up and try again later
        return failure();

      // pred is already scheduled
      auto predOpr = *prob.getLinkedOperatorType(pred);
      startTime = std::max(startTime, *predStart + *prob.getLatency(predOpr));
    }

    prob.setStartTime(op, startTime);
    return success();
  });
}
