//===- SVTraceIVerilog.cpp - Generator Callout Pass -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass adds the necessary instrumentation to a HWModule to trigger
// tracing in an iverilog simulation.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"

using namespace circt;
using namespace sv;
using namespace hw;

//===----------------------------------------------------------------------===//
// SVTraceIVerilogPass
//===----------------------------------------------------------------------===//

namespace {

struct SVTraceIVerilogPass
    : public sv::SVTraceIVerilogBase<SVTraceIVerilogPass> {
  void runOnOperation() override;
};

} // end anonymous namespace

void SVTraceIVerilogPass::runOnOperation() {
  mlir::ModuleOp mod = getOperation();

  if (topOnly) {
    auto &graph = getAnalysis<InstanceGraph>();
    auto topLevelNodes = graph.getInferredTopLevelNodes();
    if (failed(topLevelNodes) || topLevelNodes->size() != 1) {
      mod.emitError("Expected exactly one top level node");
      return signalPassFailure();
    }
    hw::HWModuleOp top =
        dyn_cast_or_null<hw::HWModuleOp>(*topLevelNodes->front()->getModule());
    if (!top) {
      mod.emitError("top module is not a HWModuleOp");
      return signalPassFailure();
    }
    targetModuleName.setValue(top.getName().str());
  }

  for (auto hwmod : mod.getOps<hw::HWModuleOp>()) {
    if (!targetModuleName.empty() &&
        hwmod.getName() != targetModuleName.getValue())
      continue;
    OpBuilder builder(hwmod.getBodyBlock(), hwmod.getBodyBlock()->begin());
    std::string traceMacro;
    llvm::raw_string_ostream ss(traceMacro);
    auto modName = hwmod.getName();
    ss << "initial begin\n  $dumpfile (\"" << directoryName.getValue()
       << modName << ".vcd\");\n  $dumpvars (0, " << modName
       << ");\n  #1;\nend\n";
    builder.create<sv::VerbatimOp>(hwmod.getLoc(), ss.str());
  }
}

std::unique_ptr<Pass> circt::sv::createSVTraceIVerilogPass() {
  return std::make_unique<SVTraceIVerilogPass>();
}
