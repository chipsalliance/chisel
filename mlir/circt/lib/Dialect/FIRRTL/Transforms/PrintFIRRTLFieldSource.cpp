//===- PrintFIRRTLFieldSource.cpp - Print the Field Source ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Print the FieldSource analysis.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLFieldSource.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "llvm/Support/raw_ostream.h"

using namespace circt;
using namespace firrtl;

namespace {
struct PrintFIRRTLFieldSourcePass
    : public PrintFIRRTLFieldSourcePassBase<PrintFIRRTLFieldSourcePass> {
  PrintFIRRTLFieldSourcePass(raw_ostream &os) : os(os) {}

  void visitValue(const FieldSource &fieldRefs, Value v) {
    auto *p = fieldRefs.nodeForValue(v);
    if (p) {
      if (p->isRoot())
        os << "ROOT: ";
      else
        os << "SUB:  " << v;
      os << " : " << p->src << " : {";
      llvm::interleaveComma(p->path, os);
      os << "} ";
      os << ((p->flow == Flow::Source) ? "Source"
             : (p->flow == Flow::Sink) ? "Sink"
                                       : "Duplex");
      if (p->isSrcWritable())
        os << " writable";
      if (p->isSrcTransparent())
        os << " transparent";
      os << "\n";
    }
  }

  void visitOp(const FieldSource &fieldRefs, Operation *op) {
    for (auto r : op->getResults())
      visitValue(fieldRefs, r);
    // recurse in to regions
    for (auto &r : op->getRegions())
      for (auto &b : r.getBlocks())
        for (auto &op : b)
          visitOp(fieldRefs, &op);
  }

  void runOnOperation() override {
    auto modOp = getOperation();
    os << "** " << modOp.getName() << "\n";
    auto &fieldRefs = getAnalysis<FieldSource>();
    for (auto port : modOp.getBodyBlock()->getArguments())
      visitValue(fieldRefs, port);
    for (auto &op : *modOp.getBodyBlock())
      visitOp(fieldRefs, &op);

    markAllAnalysesPreserved();
  }
  raw_ostream &os;
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> circt::firrtl::createFIRRTLFieldSourcePass() {
  return std::make_unique<PrintFIRRTLFieldSourcePass>(llvm::errs());
}
