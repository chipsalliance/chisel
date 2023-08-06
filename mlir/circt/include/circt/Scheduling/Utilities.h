//===- Utilities.h - Library of scheduling utilities ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a library of scheduling utilities.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SCHEDULING_UTILITIES_H
#define CIRCT_SCHEDULING_UTILITIES_H

#include "circt/Scheduling/Problems.h"

#include "llvm/Support/raw_ostream.h"

#include <functional>

namespace circt {
namespace scheduling {

using HandleOpFn = std::function<LogicalResult(Operation *)>;
/// Visit \p prob's operations in topological order, using an internal worklist.
///
/// \p fun is expected to report success if the given operation was handled
/// successfully, and failure if an unhandled predecessor was detected.
///
/// Fails if the dependence graph contains cycles.
LogicalResult handleOperationsInTopologicalOrder(Problem &prob, HandleOpFn fun);

/// Analyse combinational chains in \p prob's dependence graph and determine
/// pairs of operations that must be separated by at least one time step in
/// order to prevent the accumulated delays exceeding the given \p cycleTime.
/// The dependences in the \p result vector require special handling in the
/// concrete scheduling algorithm.
///
/// Fails if \p prob contains operator types with incoming/outgoing delays
/// greater than \p cycleTime, or if the dependence graph contains cycles.
LogicalResult
computeChainBreakingDependences(ChainingProblem &prob, float cycleTime,
                                SmallVectorImpl<Problem::Dependence> &result);

/// Assuming \p prob is scheduled and contains (integer) start times, this
/// method fills in the start times in cycle in an ASAP fashion.
///
/// Fails if the dependence graph contains cycles.
LogicalResult computeStartTimesInCycle(ChainingProblem &prob);

/// Export \p prob as a DOT graph into \p fileName.
void dumpAsDOT(Problem &prob, StringRef fileName);

/// Print \p prob as a DOT graph onto \p stream.
void dumpAsDOT(Problem &prob, raw_ostream &stream);

} // namespace scheduling
} // namespace circt

#endif // CIRCT_SCHEDULING_UTILITIES_H
