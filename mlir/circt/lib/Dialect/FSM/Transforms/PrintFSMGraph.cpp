//===- FSMPrintFSMGraph.cpp - Print the instance graph ------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Print the module hierarchy.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FSM/FSMGraph.h"
#include "circt/Dialect/FSM/FSMPasses.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"

using namespace circt;
using namespace fsm;

namespace {
struct PrintFSMGraphPass : public PrintFSMGraphBase<PrintFSMGraphPass> {
  PrintFSMGraphPass(raw_ostream &os) : os(os) {}
  void runOnOperation() override {
    getOperation().walk([&](fsm::MachineOp machine) {
      auto fsmGraph = fsm::FSMGraph(machine);
      llvm::WriteGraph(os, &fsmGraph, /*ShortNames=*/false);
    });
  }
  raw_ostream &os;
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> circt::fsm::createPrintFSMGraphPass() {
  return std::make_unique<PrintFSMGraphPass>(llvm::errs());
}
