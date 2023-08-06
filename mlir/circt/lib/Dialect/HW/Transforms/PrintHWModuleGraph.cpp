//===- PrintHWModuleGraph.cpp - Print the instance graph --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Prints an HW module as a .dot graph.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/HW/HWModuleGraph.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"

using namespace circt;
using namespace hw;

namespace {
struct PrintHWModuleGraphPass
    : public PrintHWModuleGraphBase<PrintHWModuleGraphPass> {
  PrintHWModuleGraphPass(raw_ostream &os) : os(os) {}
  void runOnOperation() override {
    getOperation().walk([&](hw::HWModuleOp module) {
      // We don't really have any other way of forwarding draw arguments to the
      // DOTGraphTraits for HWModule except through the module itself - as an
      // attribute.
      module->setAttr("dot_verboseEdges",
                      BoolAttr::get(module.getContext(), verboseEdges));

      llvm::WriteGraph(os, module, /*ShortNames=*/false);
    });
  }
  raw_ostream &os;
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> circt::hw::createPrintHWModuleGraphPass() {
  return std::make_unique<PrintHWModuleGraphPass>(llvm::errs());
}
