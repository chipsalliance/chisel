//===- LowerMatches.cpp - Lower match statements to whens -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LowerMatchesPass, which lowers match statements to when
// statements
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace circt;
using namespace firrtl;

namespace {
class LowerMatchesPass : public LowerMatchesBase<LowerMatchesPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

static void lowerMatch(MatchOp match) {

  // If this is an empty enumeration statement, just delete the match.
  auto numCases = match->getNumRegions();
  if (!numCases) {
    match->erase();
    return;
  }

  ImplicitLocOpBuilder b(match.getLoc(), match);
  auto input = match.getInput();
  for (size_t i = 0; i < numCases - 1; ++i) {
    // Create a WhenOp which tests the enum's tag.
    auto condition = b.create<IsTagOp>(input, match.getFieldIndexAttr(i));
    auto when = b.create<WhenOp>(condition, /*withElse=*/true);

    // Move the case block to the WhenOp.
    auto *thenBlock = &when.getThenBlock();
    auto *caseBlock = &match.getRegion(i).front();
    caseBlock->moveBefore(thenBlock);
    thenBlock->erase();

    // Replace the block argument with a subtag op.
    b.setInsertionPointToStart(caseBlock);
    auto data = b.create<SubtagOp>(input, match.getFieldIndexAttr(i));
    caseBlock->getArgument(0).replaceAllUsesWith(data);
    caseBlock->eraseArgument(0);

    // Change insertion to the else block.
    b.setInsertionPointToStart(&when.getElseBlock());
  }

  // At this point, the insertion point is either in the final else-block, or
  // if there was only 1 variant, right before the match operation.

  // Replace the block argument with a subtag op.
  auto data = b.create<SubtagOp>(input, match.getFieldIndexAttr(numCases - 1));

  // Get the final block from the match statement, and splice it into the
  // current insertion point.
  auto *caseBlock = &match->getRegions().back().front();
  caseBlock->getArgument(0).replaceAllUsesWith(data);
  auto *defaultBlock = b.getInsertionBlock();
  defaultBlock->getOperations().splice(b.getInsertionPoint(),
                                       caseBlock->getOperations());
  match->erase();
}

void LowerMatchesPass::runOnOperation() {
  getOperation()->walk(&lowerMatch);
  markAnalysesPreserved<InstanceGraph>();
}

std::unique_ptr<mlir::Pass> circt::firrtl::createLowerMatchesPass() {
  return std::make_unique<LowerMatchesPass>();
}
