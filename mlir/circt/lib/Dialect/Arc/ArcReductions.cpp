//===- ArcReductions.cpp - Reduction patterns for the Arc Dialect -=-------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcReductions.h"
#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arc-reductions"

using namespace circt;
using namespace arc;

//===----------------------------------------------------------------------===//
// Reduction patterns
//===----------------------------------------------------------------------===//

/// A sample reduction pattern that converts `arc.state` operations to the
/// simpler `arc.call` operation and removes clock, latency, name attributes,
/// enables, and resets in the process.
struct StateElimination : public OpReduction<StateOp> {
  LogicalResult rewrite(StateOp stateOp) override {
    OpBuilder builder(stateOp);
    ValueRange results =
        builder
            .create<arc::CallOp>(stateOp.getLoc(), stateOp->getResultTypes(),
                                 stateOp.getArcAttr(), stateOp.getInputs())
            ->getResults();
    stateOp.replaceAllUsesWith(results);
    stateOp.erase();
    return success();
  }

  std::string getName() const override { return "arc-state-elimination"; }
};

//===----------------------------------------------------------------------===//
// Reduction Registration
//===----------------------------------------------------------------------===//

void ArcReducePatternDialectInterface::populateReducePatterns(
    circt::ReducePatternSet &patterns) const {
  // Gather a list of reduction patterns that we should try. Ideally these are
  // assigned reasonable benefit indicators (higher benefit patterns are
  // prioritized). For example, things that can knock out entire modules while
  // being cheap should be tried first (and thus have higher benefit), before
  // trying to tweak operands of individual arithmetic ops.
  patterns.add<PassReduction, 4>(getContext(), arc::createStripSVPass(), true,
                                 true);
  patterns.add<PassReduction, 3>(getContext(), arc::createDedupPass());
  patterns.add<StateElimination, 2>();
  patterns.add<PassReduction, 1>(getContext(),
                                 arc::createArcCanonicalizerPass());
}

void arc::registerReducePatternDialectInterface(
    mlir::DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, ArcDialect *dialect) {
    dialect->addInterfaces<ArcReducePatternDialectInterface>();
  });
}
