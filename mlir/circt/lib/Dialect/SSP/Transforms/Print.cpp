//===- Print.cpp - Print pass -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the Print (as a DOT graph) pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"

#include "circt/Scheduling/Utilities.h"

using namespace circt;
using namespace scheduling;
using namespace ssp;

namespace {
struct PrintPass : public PrintBase<PrintPass> {
  explicit PrintPass(raw_ostream &os) : os(os) {}
  void runOnOperation() override;
  raw_ostream &os;
};
} // end anonymous namespace

template <typename ProblemT>
static void printInstance(InstanceOp instOp, raw_ostream &os) {
  auto prob = loadProblem<ProblemT>(instOp);
  dumpAsDOT(prob, os);
}

void PrintPass::runOnOperation() {
  auto moduleOp = getOperation();
  for (auto instOp : moduleOp.getOps<InstanceOp>()) {
    StringRef probName = instOp.getProblemName();
    if (probName.equals("Problem"))
      printInstance<Problem>(instOp, os);
    else if (probName.equals("CyclicProblem"))
      printInstance<CyclicProblem>(instOp, os);
    else if (probName.equals("ChainingProblem"))
      printInstance<ChainingProblem>(instOp, os);
    else if (probName.equals("SharedOperatorsProblem"))
      printInstance<SharedOperatorsProblem>(instOp, os);
    else if (probName.equals("ModuloProblem"))
      printInstance<ModuloProblem>(instOp, os);
    else {
      auto instName = instOp.getSymName().value_or("unnamed");
      llvm::errs() << "ssp-print-instance: Unknown problem class '" << probName
                   << "' in instance '" << instName << "'\n";
      return signalPassFailure();
    }
  }
}

std::unique_ptr<mlir::Pass> circt::ssp::createPrintPass() {
  return std::make_unique<PrintPass>(llvm::errs());
}
