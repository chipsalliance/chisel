//===- ApplyLoweringOptions.cpp - Test pass for adding lowering options ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pass for testing purposes to apply lowering options ot a module.
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "circt/Conversion/ExportVerilog.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Support/LoweringOptions.h"

using namespace circt;

namespace {

struct TestApplyLoweringOptionPass
    : public TestApplyLoweringOptionBase<TestApplyLoweringOptionPass> {
  TestApplyLoweringOptionPass() = default;
  void runOnOperation() override {
    if (!options.hasValue()) {
      markAllAnalysesPreserved();
      return;
    }
    LoweringOptions opts(options, [this](llvm::Twine tw) {
      getOperation().emitError(tw);
      signalPassFailure();
    });
    opts.setAsAttribute(getOperation());
  }
};
} // namespace

std::unique_ptr<mlir::Pass> circt::createTestApplyLoweringOptionPass() {
  return std::make_unique<TestApplyLoweringOptionPass>();
}
