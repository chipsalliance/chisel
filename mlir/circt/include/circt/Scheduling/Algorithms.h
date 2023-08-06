//===- Algorithms.h - Library of scheduling algorithms ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a library of scheduling algorithms.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SCHEDULING_ALGORITHMS_H
#define CIRCT_SCHEDULING_ALGORITHMS_H

#include "circt/Scheduling/Problems.h"

namespace circt {
namespace scheduling {

/// This is a simple list scheduler for solving the basic scheduling problem.
/// Its objective is to assign each operation its earliest possible start time,
/// or in other words, to schedule each operation as soon as possible (hence the
/// name). Fails if the dependence graph contains cycles.
LogicalResult scheduleASAP(Problem &prob);

/// Solve the basic problem using linear programming and a handwritten
/// implementation of the simplex algorithm. The objective is to minimize the
/// start time of the given \p lastOp. Fails if the dependence graph contains
/// cycles, or \p prob does not include \p lastOp.
LogicalResult scheduleSimplex(Problem &prob, Operation *lastOp);

/// Solve the resource-free cyclic problem using linear programming and a
/// handwritten implementation of the simplex algorithm. The objectives are to
/// determine the smallest feasible initiation interval, and to minimize the
/// start time of the given \p lastOp. Fails if the dependence graph contains
/// cycles that do not include at least one edge with a non-zero distance, or
/// \p prob does not include \p lastOp.
LogicalResult scheduleSimplex(CyclicProblem &prob, Operation *lastOp);

/// Solve the acyclic problem with shared operators using a linear
/// programming-based heuristic. The approach tries to minimize the start time
/// of the given \p lastOp, but optimality is not guaranteed. Fails if the
/// dependence graph contains cycles, or \p prob does not include \p lastOp.
LogicalResult scheduleSimplex(SharedOperatorsProblem &prob, Operation *lastOp);

/// Solve the modulo scheduling problem using a linear programming-based
/// heuristic. The approach tries to determine the smallest feasible initiation
/// interval, and to minimize the start time of the given \p lastOp, but
/// optimality is not guaranteed. Fails if the dependence graph contains cycles
/// that do not include at least one edge with a non-zero distance, \p prob
/// does not include \p lastOp, or \p lastOp is not the unique sink of the
/// dependence graph.
LogicalResult scheduleSimplex(ModuloProblem &prob, Operation *lastOp);

/// Solve the acyclic, chaining-enabled problem using linear programming and a
/// handwritten implementation of the simplex algorithm. This approach strictly
/// adheres to the given maximum \p cycleTime. The objective is to minimize the
/// start time of the given \p lastOp. Fails if the dependence graph contains
/// cycles, or individual operator types have delays larger than \p cycleTime,
/// or \p prob does not include \p lastOp.
LogicalResult scheduleSimplex(ChainingProblem &prob, Operation *lastOp,
                              float cycleTime);

/// Solve the basic problem using linear programming and an external LP solver.
/// The objective is to minimize the start time of the given \p lastOp. Fails if
/// the dependence graph contains cycles, or \p prob does not include \p lastOp.
LogicalResult scheduleLP(Problem &prob, Operation *lastOp);

/// Solve the resource-free cyclic problem using integer linear programming and
/// an external ILP solver. The objectives are to determine the smallest
/// feasible initiation interval, and to minimize the start time of the given \p
/// lastOp. Fails if the dependence graph contains cycles that do not include at
/// least one edge with a non-zero distance, or \p prob does not include
/// \p lastOp.
LogicalResult scheduleLP(CyclicProblem &prob, Operation *lastOp);

/// Solve the acyclic problem with shared operators using constraint programming
/// and an external SAT solver. The objective is to minimize the start time of
/// the given \p lastOp. Fails if the dependence graph contains cycles, or \p
/// prob does not include \p lastOp.
LogicalResult scheduleCPSAT(SharedOperatorsProblem &prob, Operation *lastOp);

} // namespace scheduling
} // namespace circt

#endif // CIRCT_SCHEDULING_ALGORITHMS_H
