//===- SchedulingAnalysis.h - scheduling analyses -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for methods that perform analysis
// involving scheduling.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_ANALYSIS_SCHEDULING_ANALYSIS_H
#define CIRCT_ANALYSIS_SCHEDULING_ANALYSIS_H

#include "circt/Analysis/DependenceAnalysis.h"
#include "circt/Scheduling/Problems.h"
#include "circt/Support/LLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"

namespace mlir {
class AnalysisManager;
namespace func {
class FuncOp;
} // namespace func
} // namespace mlir

using namespace mlir;
using namespace circt::scheduling;

namespace circt {
namespace analysis {

/// CyclicSchedulingAnalysis constructs a CyclicProblem for each AffineForOp by
/// performing a memory dependence analysis and inserting dependences into the
/// problem. The client should retrieve the partially complete problem to add
/// and associate operator types.
struct CyclicSchedulingAnalysis {
  CyclicSchedulingAnalysis(Operation *funcOp, AnalysisManager &am);

  CyclicProblem &getProblem(affine::AffineForOp forOp);

private:
  void analyzeForOp(affine::AffineForOp forOp,
                    MemoryDependenceAnalysis memoryAnalysis);

  DenseMap<Operation *, CyclicProblem> problems;
};

} // namespace analysis
} // namespace circt

#endif // CIRCT_ANALYSIS_SCHEDULING_ANALYSIS_H
