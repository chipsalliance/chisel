//===- SimplifyVariadicOps.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arc-simplify-variadic-ops"

using namespace mlir;
using namespace circt;
using namespace arc;
using namespace hw;

namespace {
struct SimplifyVariadicOpsPass
    : public SimplifyVariadicOpsBase<SimplifyVariadicOpsPass> {
  SimplifyVariadicOpsPass() = default;
  SimplifyVariadicOpsPass(const SimplifyVariadicOpsPass &pass)
      : SimplifyVariadicOpsPass() {}

  void runOnOperation() override;
  void simplifyOp(Operation *op);
};
} // namespace

void SimplifyVariadicOpsPass::runOnOperation() {
  SmallVector<Operation *> opsToProcess;
  getOperation().walk([&](Operation *op) {
    if (op->hasTrait<OpTrait::IsCommutative>() && op->getNumRegions() == 0 &&
        op->getNumSuccessors() == 0 && op->getNumResults() == 1 &&
        op->getNumOperands() > 2 && isMemoryEffectFree(op))
      opsToProcess.push_back(op);
  });
  for (auto *op : opsToProcess)
    simplifyOp(op);
}

void SimplifyVariadicOpsPass::simplifyOp(Operation *op) {
  // Gather the list of operands together with the defining op. Block arguments
  // simply get no op assigned. This is also where we bail out if the block
  // argument or any of the defining ops is in a different block than the op
  // itself.
  auto *block = op->getBlock();
  SmallVector<Value> operands;
  for (auto operand : op->getOperands()) {
    if (auto blockArg = operand.dyn_cast<BlockArgument>()) {
      if (blockArg.getOwner() != block) {
        ++numOpsSkippedMultipleBlocks;
        return;
      }
    } else {
      auto *defOp = operand.getDefiningOp();
      if (defOp->getBlock() != block) {
        ++numOpsSkippedMultipleBlocks;
        return;
      }
    }
    operands.push_back(operand);
  }
  LLVM_DEBUG(llvm::dbgs() << "Simplifying " << *op << "\n");

  // Sort the list of operands based on the order in which their defining ops
  // appear in the block.
  llvm::sort(operands, [](auto a, auto b) {
    // Sort block args by the arg number.
    auto aBlockArg = a.template dyn_cast<BlockArgument>();
    auto bBlockArg = b.template dyn_cast<BlockArgument>();
    if (aBlockArg && bBlockArg)
      return aBlockArg.getArgNumber() < bBlockArg.getArgNumber();

    // Sort other values by block order of the defining op.
    auto *aOp = a.getDefiningOp();
    auto *bOp = b.getDefiningOp();
    if (!aOp)
      return true;
    if (!bOp)
      return false;
    return aOp->isBeforeInBlock(bOp);
  });
  LLVM_DEBUG(for (auto value
                  : operands) llvm::dbgs()
                 << "- " << value << "\n";);

  // Keep some statistics whether we actually did do some reordering.
  for (auto [a, b] : llvm::zip(operands, op->getOperands())) {
    if (a != b) {
      ++numOpsReordered;
      break;
    }
  }

  // Split up the variadic operation by going through the operands and creating
  // pairwise versions of the op as close as possible to the operands.
  Value reduced = operands[0];
  auto builder = OpBuilder::atBlockBegin(block);
  for (auto value : llvm::drop_begin(operands)) {
    if (auto *defOp = value.getDefiningOp())
      builder.setInsertionPointAfter(defOp);
    reduced = builder
                  .create(op->getLoc(), op->getName().getIdentifier(),
                          ValueRange{reduced, value}, op->getResultTypes(),
                          op->getAttrs())
                  ->getResult(0);
    ++numOpsCreated;
  }
  op->getResult(0).replaceAllUsesWith(reduced);
  op->erase();
  ++numOpsSimplified;
}

std::unique_ptr<Pass> arc::createSimplifyVariadicOpsPass() {
  return std::make_unique<SimplifyVariadicOpsPass>();
}
