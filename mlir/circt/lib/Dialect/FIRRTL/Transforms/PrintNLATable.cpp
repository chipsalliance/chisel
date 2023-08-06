//===- PrintNLATable.cpp - Print the NLA Table ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Print the NLA Table.  This is primarily a debugging pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/NLATable.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "llvm/Support/raw_ostream.h"

using namespace circt;
using namespace firrtl;

namespace {
struct PrintNLATablePass : public PrintNLATableBase<PrintNLATablePass> {
  PrintNLATablePass(raw_ostream &os) : os(os) {}
  void runOnOperation() override {
    auto circuitOp = getOperation();
    auto &nlaTable = getAnalysis<NLATable>();
    markAllAnalysesPreserved();

    for (auto &mop : *cast<CircuitOp>(circuitOp).getBodyBlock()) {
      auto mod = dyn_cast<FModuleLike>(mop);
      if (!mod)
        continue;
      os << mod.getModuleName() << ": ";
      for (auto nla : nlaTable.lookup(mod))
        os << nla.getSymName() << ", ";
      os << '\n';
    }
  }
  raw_ostream &os;
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> circt::firrtl::createPrintNLATablePass() {
  return std::make_unique<PrintNLATablePass>(llvm::errs());
}
