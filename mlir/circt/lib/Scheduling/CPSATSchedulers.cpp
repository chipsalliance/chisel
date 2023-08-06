//===- CPSATSchedulers.cpp - Schedulers using external CPSAT solvers
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of cp-sat programming-based schedulers using external solvers
// via OR-Tools.
//
//===----------------------------------------------------------------------===//

#include "circt/Scheduling/Algorithms.h"

#include "mlir/IR/Operation.h"

#include "ortools/sat/cp_model.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"

#define DEBUG_TYPE "cpsat-schedulers"

using namespace circt;
using namespace circt::scheduling;
using namespace operations_research;
using namespace operations_research::sat;

using llvm::dbgs;
using llvm::format;

/// Solve the shared operators problem by modeling it as a Resource
/// Constrained Project Scheduling Problem (RCPSP), which in turn is formulated
/// as a Constraint Programming (CP) Satisfiability (SAT) problem.
///
/// This is a high-fidelity translation of
/// https://github.com/google/or-tools/blob/stable/examples/python/rcpsp_sat.py
/// but a gentler introduction (though with differing formulation) is
/// https://python-mip.readthedocs.io/en/latest/examples.html
LogicalResult scheduling::scheduleCPSAT(SharedOperatorsProblem &prob,
                                        Operation *lastOp) {
  Operation *containingOp = prob.getContainingOp();
  if (!prob.hasOperation(lastOp))
    return containingOp->emitError("problem does not include last operation");

  CpModelBuilder cpModel;
  auto &tasks = prob.getOperations();

  DenseMap<Operation *, IntVar> taskStarts;
  DenseMap<Operation *, IntVar> taskEnds;
  DenseMap<Problem::OperatorType, SmallVector<IntervalVar, 4>>
      resourcesToTaskIntervals;

  // First get horizon, i.e., the time taken if all operations were executed
  // sequentially.
  unsigned horizon = 0;
  for (auto *task : tasks) {
    unsigned duration = *prob.getLatency(*prob.getLinkedOperatorType(task));
    horizon += duration;
  }

  // Build task-interval decision variables, which effectively serve to
  // constrain startVar and endVar to be duration apart. Then map them
  // to the resources (operators) consumed during those intervals. Note,
  // resources are in fact not constrained to be occupied for the whole of the
  // task interval, but only during the first "tick". See comment below
  // regarding cpModel.NewFixedSizeIntervalVar.
  for (auto item : llvm::enumerate(tasks)) {
    auto i = item.index();
    auto *task = item.value();
    IntVar startVar = cpModel.NewIntVar(Domain(0, horizon))
                          .WithName((Twine("start_of_task_") + Twine(i)).str());
    IntVar endVar = cpModel.NewIntVar(Domain(0, horizon))
                        .WithName((Twine("end_of_task_") + Twine(i)).str());
    taskStarts[task] = startVar;
    taskEnds[task] = endVar;
    auto resource = prob.getLinkedOperatorType(task);
    unsigned duration = *prob.getLatency(*resource);
    IntervalVar taskInterval =
        cpModel.NewIntervalVar(startVar, duration, endVar)
            .WithName((Twine("task_interval_") + Twine(i)).str());

    if (prob.getLimit(*resource))
      resourcesToTaskIntervals[resource.value()].emplace_back(taskInterval);
  }

  // Check for cycles and otherwise establish operation ordering
  // constraints.
  for (Operation *task : tasks) {
    for (auto dep : prob.getDependences(task)) {
      Operation *src = dep.getSource();
      Operation *dst = dep.getDestination();
      if (src == dst)
        return containingOp->emitError() << "dependence cycle detected";
      cpModel.AddLessOrEqual(taskEnds[src], taskStarts[dst]);
    }
  }

  // Establish "cumulative" constraints in order to constrain maximum
  // concurrent usage of operators.
  for (auto resourceToTaskIntervals : resourcesToTaskIntervals) {
    Problem::OperatorType &resource = resourceToTaskIntervals.getFirst();
    auto capacity = prob.getLimit(resource);
    SmallVector<IntervalVar, 4> &taskIntervals =
        resourceToTaskIntervals.getSecond();
    // The semantics of cumulative constraints in or-tools are such that
    // for any integer point, the sum of the demands across all
    // intervals containing that point does not exceed the capacity of the
    // resource. Thus tasks, in 1-1 correspondence with their intervals, are
    // constrained to satisfy maximum resource requirements.
    // See https://or.stackexchange.com/a/3363 for more details.
    CumulativeConstraint cumu = cpModel.AddCumulative(capacity.value());
    for (const auto &item : llvm::enumerate(taskIntervals)) {
      auto i = item.index();
      auto taskInterval = item.value();
      IntVar demandVar = cpModel.NewIntVar(Domain(1)).WithName(
          (Twine("demand_") + Twine(i) + Twine("_") + Twine(resource.strref()))
              .str());
      // Conventional formulation for SharedOperatorsProblem;
      // interval during which the resource is occupied has size 1.
      IntervalVar start =
          cpModel.NewFixedSizeIntervalVar(taskInterval.StartExpr(), 1);
      cumu.AddDemand(start, demandVar);
    }
  }

  cpModel.Minimize(taskEnds[lastOp]);

  Model model;

  int numSolutions = 0;
  model.Add(NewFeasibleSolutionObserver([&](const CpSolverResponse &r) {
    LLVM_DEBUG(dbgs() << "Solution " << numSolutions << '\n');
    LLVM_DEBUG(dbgs() << "Solution status" << r.status() << '\n');
    ++numSolutions;
  }));

  LLVM_DEBUG(dbgs() << "Starting solver\n");
  const CpSolverResponse response = SolveCpModel(cpModel.Build(), &model);

  if (response.status() == CpSolverStatus::OPTIMAL ||
      response.status() == CpSolverStatus::FEASIBLE) {
    for (auto *task : tasks)
      prob.setStartTime(task, SolutionIntegerValue(response, taskStarts[task]));

    return success();
  }
  return containingOp->emitError() << "infeasible";
}
