//===- Roundtrip.cpp - Roundtrip pass ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the Roundtrip pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"

using namespace circt;
using namespace scheduling;
using namespace ssp;

template <typename ProblemT>
static InstanceOp roundtripAs(InstanceOp instOp, bool check, bool verify,
                              OpBuilder &builder) {
  auto prob = loadProblem<ProblemT>(instOp);

  if (check && failed(prob.check()))
    return {};
  if (verify && failed(prob.verify()))
    return {};

  return saveProblem<ProblemT>(prob, builder);
}

static InstanceOp roundtrip(InstanceOp instOp, bool check, bool verify,
                            OpBuilder &builder) {
  auto problemName = instOp.getProblemName();

  if (problemName.equals("Problem"))
    return roundtripAs<Problem>(instOp, check, verify, builder);
  if (problemName.equals("CyclicProblem"))
    return roundtripAs<CyclicProblem>(instOp, check, verify, builder);
  if (problemName.equals("SharedOperatorsProblem"))
    return roundtripAs<SharedOperatorsProblem>(instOp, check, verify, builder);
  if (problemName.equals("ModuloProblem"))
    return roundtripAs<ModuloProblem>(instOp, check, verify, builder);
  if (problemName.equals("ChainingProblem"))
    return roundtripAs<ChainingProblem>(instOp, check, verify, builder);

  llvm::errs() << "ssp-roundtrip: Unknown problem '" << problemName << "'\n";
  return {};
}

namespace {
struct RoundtripPass : public RoundtripBase<RoundtripPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

void RoundtripPass::runOnOperation() {
  auto moduleOp = getOperation();

  // Solution constraint verification implies checking the input constraints.
  bool check = checkInputConstraints || verifySolutionConstraints;
  bool verify = verifySolutionConstraints;

  SmallVector<InstanceOp> instanceOps;
  OpBuilder builder(&getContext());
  for (auto instOp : moduleOp.getOps<InstanceOp>()) {
    builder.setInsertionPoint(instOp);
    auto newInstOp = roundtrip(instOp, check, verify, builder);
    if (!newInstOp)
      return signalPassFailure();
    instanceOps.push_back(instOp);
  }

  llvm::for_each(instanceOps, [](InstanceOp op) { op.erase(); });
}

std::unique_ptr<mlir::Pass> circt::ssp::createRoundtripPass() {
  return std::make_unique<RoundtripPass>();
}
