//===- Passes.h - Helpers for pipeline instrumentation ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_PASSES_H
#define CIRCT_SUPPORT_PASSES_H

#include "circt/Support/LLVM.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "llvm/Support/Chrono.h"
#include "llvm/Support/Format.h"

namespace circt {
// This class prints logs before and after of pass executions when its pass
// operation is in `LoggedOpTypes`. Note that `runBeforePass` and `runAfterPass`
// are not thread safe so `LoggedOpTypes` must be a set of operations whose
// passes are ran sequentially (e.g. mlir::ModuleOp, firrtl::CircuitOp).
template <class... LoggedOpTypes>
class VerbosePassInstrumentation : public mlir::PassInstrumentation {
  // This stores start time points of passes.
  using TimePoint = llvm::sys::TimePoint<>;
  llvm::SmallVector<TimePoint> timePoints;
  int level = 0;
  const char *toolName;

public:
  VerbosePassInstrumentation(const char *toolName) : toolName(toolName){};
  void runBeforePass(Pass *pass, Operation *op) override {
    if (isa<LoggedOpTypes...>(op)) {
      timePoints.push_back(TimePoint::clock::now());
      auto &os = llvm::errs();
      os << llvm::format("[%s] ", toolName);
      os.indent(2 * level++);
      os << "Running \"";
      pass->printAsTextualPipeline(llvm::errs());
      os << "\"\n";
    }
  }

  void runAfterPass(Pass *pass, Operation *op) override {
    using namespace std::chrono;
    if (isa<LoggedOpTypes...>(op)) {
      auto &os = llvm::errs();
      auto elapsed = duration<double>(TimePoint::clock::now() -
                                      timePoints.pop_back_val()) /
                     seconds(1);
      os << llvm::format("[%s] ", toolName);
      os.indent(2 * --level);
      os << "-- Done in " << llvm::format("%.3f", elapsed) << " sec\n";
    }
  }
};

/// Create a simple canonicalizer pass.
std::unique_ptr<Pass> createSimpleCanonicalizerPass();

} // namespace circt

#endif // CIRCT_SUPPORT_PASSES_H
