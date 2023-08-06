//===- LPSchedulers.cpp - Schedulers using external LP solvers ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of linear programming-based schedulers using external solvers
// via OR-Tools.
//
//===----------------------------------------------------------------------===//

#include "circt/Scheduling/Algorithms.h"

#include "mlir/IR/Operation.h"

#include "ortools/linear_solver/linear_solver.h"

using namespace circt;
using namespace circt::scheduling;
using namespace operations_research;

LogicalResult scheduling::scheduleLP(Problem &prob, Operation *lastOp) {
  Operation *containingOp = prob.getContainingOp();
  if (!prob.hasOperation(lastOp))
    return containingOp->emitError("problem does not include last operation");

  MPSolver::OptimizationProblemType problemType;
  if (!MPSolver::ParseSolverType("GLOP_LINEAR_PROGRAMMING", &problemType) ||
      !MPSolver::SupportsProblemType(problemType))
    return containingOp->emitError("GLOP is unvailable");

  MPSolver solver("Problem", problemType);
  double infinity = solver.infinity();

  // Create start time variables.
  DenseMap<Operation *, MPVariable *> vars;
  unsigned i = 0;
  for (auto *op : prob.getOperations()) {
    vars[op] = solver.MakeNumVar(0, infinity, (Twine("t_") + Twine(i)).str());
    ++i;
  }

  // The objective is to minimize the start time of the last operation.
  MPObjective *objective = solver.MutableObjective();
  objective->SetCoefficient(vars[lastOp], 1);
  objective->SetMinimization();

  // Construct a linear constraint for each dependence.
  for (auto *op : prob.getOperations())
    for (auto dep : prob.getDependences(op)) {
      Operation *src = dep.getSource();
      Operation *dst = dep.getDestination();
      if (src == dst)
        return containingOp->emitError() << "dependence cycle detected";

      //     t_src + t.linkedOperatorType.latency <= t_dst
      // <=> 1 * t_src + -1 * t_dst <= -latency
      unsigned latency = *prob.getLatency(*prob.getLinkedOperatorType(src));
      MPConstraint *constraint =
          solver.MakeRowConstraint(-infinity, -((double)latency));
      constraint->SetCoefficient(vars[src], 1);
      constraint->SetCoefficient(vars[dst], -1);
    }

  // Invoke solver. The LP is infeasible if the scheduling problem contained
  // dependence cycles. Otherwise, we expect the result to be optimal.
  MPSolver::ResultStatus result = solver.Solve();
  if (result == MPSolver::INFEASIBLE)
    return containingOp->emitError() << "dependence cycle detected";
  assert(result == MPSolver::OPTIMAL);

  // Retrieve start times.
  for (auto *op : prob.getOperations())
    prob.setStartTime(op, std::round(vars[op]->solution_value()));

  return success();
}

LogicalResult scheduling::scheduleLP(CyclicProblem &prob, Operation *lastOp) {
  Operation *containingOp = prob.getContainingOp();
  if (!prob.hasOperation(lastOp))
    return containingOp->emitError("problem does not include last operation");

  MPSolver::OptimizationProblemType probType;
  if (!MPSolver::ParseSolverType("CBC_MIXED_INTEGER_PROGRAMMING", &probType) ||
      !MPSolver::SupportsProblemType(probType))
    return containingOp->emitError("Cbc is unavailable");

  MPSolver solver("CyclicProblem", probType);
  double infinity = solver.infinity();

  // Create II variable.
  MPVariable *ii = solver.MakeIntVar(1, infinity, "II");

  // Create start time variables (and collect latencies to compute upper bound
  // for the latest start time).
  DenseMap<Operation *, MPVariable *> t;
  unsigned upperBound = 0;
  unsigned i = 0;
  for (auto *op : prob.getOperations()) {
    t[op] = solver.MakeIntVar(0, infinity, (Twine("t_") + Twine(i)).str());
    upperBound += *prob.getLatency(*prob.getLinkedOperatorType(op));
    ++i;
  }

  // The objective is to minimize the II as well as the start time of the last
  // operation. We use a weighted sum to encode both objectives in a single
  // linear expression.
  MPObjective *objective = solver.MutableObjective();
  objective->SetCoefficient(ii, upperBound);
  objective->SetCoefficient(t[lastOp], 1);
  objective->SetMinimization();

  // Construct a linear constraint for each dependence.
  for (auto *op : prob.getOperations())
    for (auto dep : prob.getDependences(op)) {
      Operation *src = dep.getSource();
      Operation *dst = dep.getDestination();

      //     t_src + t_src.linkedOperatorType.latency <= t_dst + e.distance * II
      // <=> 1 * t_src + -1 * t_dst + -e.distance * II <= -latency
      unsigned latency = *prob.getLatency(*prob.getLinkedOperatorType(src));
      MPConstraint *constraint =
          solver.MakeRowConstraint(-infinity, -((double)latency));
      if (src != dst) { // Handle self-arcs.
        constraint->SetCoefficient(t[src], 1);
        constraint->SetCoefficient(t[dst], -1);
      }
      constraint->SetCoefficient(ii,
                                 -((double)prob.getDistance(dep).value_or(0)));
    }

  // Invoke solver. The ILP is infeasible if the scheduling problem contained
  // dependence graph contains cycles that do not include at least one edge with
  // a non-zero distance. Otherwise, we expect the result to be optimal.
  MPSolver::ResultStatus result = solver.Solve();
  if (result == MPSolver::INFEASIBLE)
    return containingOp->emitError() << "dependence cycle detected";
  assert(result == MPSolver::OPTIMAL);

  // Retrieve II and start times.
  prob.setInitiationInterval(std::round(ii->solution_value()));
  for (auto *op : prob.getOperations())
    prob.setStartTime(op, std::round(t[op]->solution_value()));

  return success();
}
