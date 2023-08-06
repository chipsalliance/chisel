//===- Schedule.cpp - Schedule pass -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the Schedule pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"

#include "circt/Scheduling/Algorithms.h"

#include "llvm/ADT/StringExtras.h"

using namespace circt;
using namespace scheduling;
using namespace ssp;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Determine "last" operation, i.e. the one whose start time we are supposed to
// minimize.
static OperationOp getLastOp(InstanceOp instOp, StringRef options) {
  StringRef lastOpName = "";
  for (StringRef option : llvm::split(options, ',')) {
    if (option.consume_front("last-op-name=")) {
      lastOpName = option;
      break;
    }
  }

  auto graphOp = instOp.getDependenceGraph();
  if (lastOpName.empty() && !graphOp.getBodyBlock()->empty())
    return cast<OperationOp>(graphOp.getBodyBlock()->back());
  return graphOp.lookupSymbol<OperationOp>(lastOpName);
}

// Determine desired cycle time (only relevant for `ChainingProblem` instances).
static std::optional<float> getCycleTime(StringRef options) {
  for (StringRef option : llvm::split(options, ',')) {
    if (option.consume_front("cycle-time="))
      return std::stof(option.str());
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// ASAP scheduler
//===----------------------------------------------------------------------===//

static InstanceOp scheduleWithASAP(InstanceOp instOp, OpBuilder &builder) {
  auto problemName = instOp.getProblemName();
  if (!problemName.equals("Problem")) {
    llvm::errs() << "ssp-schedule: Unsupported problem '" << problemName
                 << "' for ASAP scheduler\n";
    return {};
  }

  auto prob = loadProblem<Problem>(instOp);
  if (failed(prob.check()) || failed(scheduling::scheduleASAP(prob)) ||
      failed(prob.verify()))
    return {};
  return saveProblem(prob, builder);
}

//===----------------------------------------------------------------------===//
// Simplex schedulers
//===----------------------------------------------------------------------===//

template <typename ProblemT>
static InstanceOp scheduleProblemTWithSimplex(InstanceOp instOp,
                                              Operation *lastOp,
                                              OpBuilder &builder) {
  auto prob = loadProblem<ProblemT>(instOp);
  if (failed(prob.check()) ||
      failed(scheduling::scheduleSimplex(prob, lastOp)) ||
      failed(prob.verify()))
    return {};
  return saveProblem(prob, builder);
}

static InstanceOp scheduleChainingProblemWithSimplex(InstanceOp instOp,
                                                     Operation *lastOp,
                                                     float cycleTime,
                                                     OpBuilder &builder) {
  auto prob = loadProblem<scheduling::ChainingProblem>(instOp);
  if (failed(prob.check()) ||
      failed(scheduling::scheduleSimplex(prob, lastOp, cycleTime)) ||
      failed(prob.verify()))
    return {};
  return saveProblem(prob, builder);
}

static InstanceOp scheduleWithSimplex(InstanceOp instOp, StringRef options,
                                      OpBuilder &builder) {
  auto lastOp = getLastOp(instOp, options);
  if (!lastOp) {
    auto instName = instOp.getSymName().value_or("unnamed");
    llvm::errs()
        << "ssp-schedule: Ambiguous objective for simplex scheduler: Instance '"
        << instName << "' has no designated last operation\n";
    return {};
  }

  auto problemName = instOp.getProblemName();
  if (problemName.equals("Problem"))
    return scheduleProblemTWithSimplex<Problem>(instOp, lastOp, builder);
  if (problemName.equals("CyclicProblem"))
    return scheduleProblemTWithSimplex<CyclicProblem>(instOp, lastOp, builder);
  if (problemName.equals("SharedOperatorsProblem"))
    return scheduleProblemTWithSimplex<SharedOperatorsProblem>(instOp, lastOp,
                                                               builder);
  if (problemName.equals("ModuloProblem"))
    return scheduleProblemTWithSimplex<ModuloProblem>(instOp, lastOp, builder);
  if (problemName.equals("ChainingProblem")) {
    if (auto cycleTime = getCycleTime(options))
      return scheduleChainingProblemWithSimplex(instOp, lastOp,
                                                cycleTime.value(), builder);
    llvm::errs() << "ssp-schedule: Missing option 'cycle-time' for "
                    "ChainingProblem simplex scheduler\n";
    return {};
  }

  llvm::errs() << "ssp-schedule: Unsupported problem '" << problemName
               << "' for simplex scheduler\n";
  return {};
}

#ifdef SCHEDULING_OR_TOOLS

//===----------------------------------------------------------------------===//
// LP schedulers (require OR-Tools)
//===----------------------------------------------------------------------===//

template <typename ProblemT>
static InstanceOp scheduleProblemTWithLP(InstanceOp instOp, Operation *lastOp,
                                         OpBuilder &builder) {
  auto prob = loadProblem<ProblemT>(instOp);
  if (failed(prob.check()) || failed(scheduling::scheduleLP(prob, lastOp)) ||
      failed(prob.verify()))
    return {};
  return saveProblem(prob, builder);
}

static InstanceOp scheduleWithLP(InstanceOp instOp, StringRef options,
                                 OpBuilder &builder) {
  auto lastOp = getLastOp(instOp, options);
  if (!lastOp) {
    auto instName = instOp.getSymName().value_or("unnamed");
    llvm::errs()
        << "ssp-schedule: Ambiguous objective for LP scheduler: Instance '"
        << instName << "' has no designated last operation\n";
    return {};
  }

  auto problemName = instOp.getProblemName();
  if (problemName.equals("Problem"))
    return scheduleProblemTWithLP<Problem>(instOp, lastOp, builder);
  if (problemName.equals("CyclicProblem"))
    return scheduleProblemTWithLP<CyclicProblem>(instOp, lastOp, builder);

  llvm::errs() << "ssp-schedule: Unsupported problem '" << problemName
               << "' for LP scheduler\n";
  return {};
}

//===----------------------------------------------------------------------===//
// CPSAT scheduler (requires OR-Tools)
//===----------------------------------------------------------------------===//

static InstanceOp scheduleWithCPSAT(InstanceOp instOp, StringRef options,
                                    OpBuilder &builder) {
  auto lastOp = getLastOp(instOp, options);
  if (!lastOp) {
    auto instName = instOp.getSymName().value_or("unnamed");
    llvm::errs()
        << "ssp-schedule: Ambiguous objective for CPSAT scheduler: Instance '"
        << instName << "' has no designated last operation\n";
    return {};
  }

  auto problemName = instOp.getProblemName();
  if (!problemName.equals("SharedOperatorsProblem")) {
    llvm::errs() << "ssp-schedule: Unsupported problem '" << problemName
                 << "' for CPSAT scheduler\n";
    return {};
  }

  auto prob = loadProblem<SharedOperatorsProblem>(instOp);
  if (failed(prob.check()) || failed(scheduling::scheduleCPSAT(prob, lastOp)) ||
      failed(prob.verify()))
    return {};
  return saveProblem(prob, builder);
}

#endif // SCHEDULING_OR_TOOLS

//===----------------------------------------------------------------------===//
// Algorithm dispatcher
//===----------------------------------------------------------------------===//

static InstanceOp scheduleWith(InstanceOp instOp, StringRef scheduler,
                               StringRef options, OpBuilder &builder) {
  if (scheduler.empty() || scheduler.equals("simplex"))
    return scheduleWithSimplex(instOp, options, builder);
  if (scheduler.equals("asap"))
    return scheduleWithASAP(instOp, builder);
#ifdef SCHEDULING_OR_TOOLS
  if (scheduler.equals("lp"))
    return scheduleWithLP(instOp, options, builder);
  if (scheduler.equals("cpsat"))
    return scheduleWithCPSAT(instOp, options, builder);
#endif

  llvm::errs() << "ssp-schedule: Unsupported scheduler '" << scheduler
               << "' requested\n";
  return {};
}

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

namespace {
struct SchedulePass : public ScheduleBase<SchedulePass> {
  void runOnOperation() override;
};
} // end anonymous namespace

void SchedulePass::runOnOperation() {
  auto moduleOp = getOperation();

  SmallVector<InstanceOp> instanceOps;
  OpBuilder builder(&getContext());
  for (auto instOp : moduleOp.getOps<InstanceOp>()) {
    builder.setInsertionPoint(instOp);
    auto scheduledOp = scheduleWith(instOp, scheduler.getValue(),
                                    schedulerOptions.getValue(), builder);
    if (!scheduledOp)
      return signalPassFailure();
    instanceOps.push_back(instOp);
  }

  llvm::for_each(instanceOps, [](InstanceOp op) { op.erase(); });
}

std::unique_ptr<mlir::Pass> circt::ssp::createSchedulePass() {
  return std::make_unique<SchedulePass>();
}
