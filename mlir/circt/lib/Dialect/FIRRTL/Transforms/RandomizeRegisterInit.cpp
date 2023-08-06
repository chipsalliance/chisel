//===- RandomizeRegisterInit.cpp - Randomize register initialization ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the RandomizeRegisterInit pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/Parallel.h"

using namespace mlir;
using namespace circt;
using namespace firrtl;

namespace {
struct RandomizeRegisterInitPass
    : public RandomizeRegisterInitBase<RandomizeRegisterInitPass> {
  void runOnOperation() override;
};

} // end anonymous namespace

std::unique_ptr<mlir::Pass> circt::firrtl::createRandomizeRegisterInitPass() {
  return std::make_unique<RandomizeRegisterInitPass>();
}

/// Create attributes indicating the required size of random initialization
/// values for each register in the module, and mark which range of these values
/// each register should consume. The goal is for registers to always read the
/// same random bits for the same seed, regardless of optimizations that might
/// remove registers.
static void createRandomizationAttributes(FModuleOp mod) {
  OpBuilder builder(mod);

  // Walk all registers.
  uint64_t currentWidth = 0;
  auto ui64Type = builder.getIntegerType(64, false);
  mod.walk([&](Operation *op) {
    if (!isa<RegOp, RegResetOp>(op))
      return;

    // Compute the width of all registers, and remember which bits are assigned
    // to each register.
    auto regType = type_cast<FIRRTLBaseType>(op->getResult(0).getType());
    std::optional<int64_t> regWidth = getBitWidth(regType);
    assert(regWidth.has_value() && "register must have a valid FIRRTL width");

    auto start = builder.getIntegerAttr(ui64Type, currentWidth);
    op->setAttr("firrtl.random_init_start", start);

    currentWidth += *regWidth;
  });
}

void RandomizeRegisterInitPass::runOnOperation() {
  createRandomizationAttributes(getOperation());
}
