//===- LowerClocksToFuncs.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arc-lower-clocks-to-funcs"

using namespace mlir;
using namespace circt;
using namespace arc;
using namespace hw;
using mlir::OpTrait::ConstantLike;

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

namespace {
struct LowerClocksToFuncsPass
    : public LowerClocksToFuncsBase<LowerClocksToFuncsPass> {
  LowerClocksToFuncsPass() = default;
  LowerClocksToFuncsPass(const LowerClocksToFuncsPass &pass)
      : LowerClocksToFuncsPass() {}

  void runOnOperation() override;
  LogicalResult lowerModel(ModelOp modelOp);
  LogicalResult lowerClock(Operation *clockOp, Value modelStorageArg,
                           OpBuilder &funcBuilder);
  LogicalResult isolateClock(Operation *clockOp, Value modelStorageArg,
                             Value clockStorageArg);

  SymbolTable *symbolTable;

  Statistic numOpsCopied{this, "ops-copied", "Ops copied into clock trees"};
  Statistic numOpsMoved{this, "ops-moved", "Ops moved into clock trees"};
};
} // namespace

void LowerClocksToFuncsPass::runOnOperation() {
  symbolTable = &getAnalysis<SymbolTable>();
  for (auto op : getOperation().getOps<ModelOp>())
    if (failed(lowerModel(op)))
      return signalPassFailure();
}

LogicalResult LowerClocksToFuncsPass::lowerModel(ModelOp modelOp) {
  LLVM_DEBUG(llvm::dbgs() << "Lowering clocks in `" << modelOp.getName()
                          << "`\n");

  // Find the clocks to extract.
  SmallVector<Operation *> clocks;
  modelOp.walk([&](Operation *op) {
    if (isa<ClockTreeOp, PassThroughOp>(op))
      clocks.push_back(op);
  });

  // Perform the actual extraction.
  OpBuilder funcBuilder(modelOp);
  for (auto *op : clocks)
    if (failed(lowerClock(op, modelOp.getBody().getArgument(0), funcBuilder)))
      return failure();

  return success();
}

LogicalResult LowerClocksToFuncsPass::lowerClock(Operation *clockOp,
                                                 Value modelStorageArg,
                                                 OpBuilder &funcBuilder) {
  LLVM_DEBUG(llvm::dbgs() << "- Lowering clock " << clockOp->getName() << "\n");
  assert((isa<ClockTreeOp, PassThroughOp>(clockOp)));

  // Add a `StorageType` block argument to the clock's body block which we are
  // going to use to pass the storage pointer to the clock once it has been
  // pulled out into a separate function.
  Region &clockRegion = clockOp->getRegion(0);
  Value clockStorageArg = clockRegion.addArgument(modelStorageArg.getType(),
                                                  modelStorageArg.getLoc());

  // Ensure the clock tree does not use any values defined outside of it.
  if (failed(isolateClock(clockOp, modelStorageArg, clockStorageArg)))
    return failure();

  // Add a return op to the end of the body.
  auto builder = OpBuilder::atBlockEnd(&clockRegion.front());
  builder.create<func::ReturnOp>(clockOp->getLoc());

  // Pick a name for the clock function.
  SmallString<32> funcName;
  funcName.append(clockOp->getParentOfType<ModelOp>().getName());
  funcName.append(isa<PassThroughOp>(clockOp) ? "_passthrough" : "_clock");
  auto funcOp = funcBuilder.create<func::FuncOp>(
      clockOp->getLoc(), funcName,
      builder.getFunctionType({modelStorageArg.getType()}, {}));
  symbolTable->insert(funcOp); // uniquifies the name
  LLVM_DEBUG(llvm::dbgs() << "  - Created function `" << funcOp.getSymName()
                          << "`\n");

  // Create a call to the function within the model.
  builder.setInsertionPoint(clockOp);
  builder.create<func::CallOp>(clockOp->getLoc(), funcOp,
                               ValueRange{modelStorageArg});

  // Move the clock's body block to the function and remove the old clock op.
  funcOp.getBody().takeBody(clockRegion);
  clockOp->erase();

  return success();
}

/// Copy any external constants that the clock tree might be using into its
/// body. Anything besides constants should no longer exist after a proper run
/// of the pipeline.
LogicalResult LowerClocksToFuncsPass::isolateClock(Operation *clockOp,
                                                   Value modelStorageArg,
                                                   Value clockStorageArg) {
  auto *clockRegion = &clockOp->getRegion(0);
  auto builder = OpBuilder::atBlockBegin(&clockRegion->front());
  DenseMap<Value, Value> copiedValues;
  auto result = clockRegion->walk([&](Operation *op) {
    for (auto &operand : op->getOpOperands()) {
      // Block arguments are okay, since there's nothing we can move.
      if (operand.get() == modelStorageArg) {
        operand.set(clockStorageArg);
        continue;
      }
      if (operand.get().isa<BlockArgument>()) {
        auto d = op->emitError(
            "operation in clock tree uses external block argument");
        d.attachNote() << "clock trees can only use external constant values";
        d.attachNote() << "see operand #" << operand.getOperandNumber();
        d.attachNote(clockOp->getLoc()) << "clock tree:";
        return WalkResult::interrupt();
      }

      // Check if the value is defined outside of the clock op.
      auto *definingOp = operand.get().getDefiningOp();
      assert(definingOp && "block arguments ruled out above");
      Region *definingRegion = definingOp->getParentRegion();
      if (clockRegion->isAncestor(definingRegion))
        continue;

      // The op is defined outside the clock, so we need to create a copy of the
      // defining inside the clock tree.
      if (auto copiedValue = copiedValues.lookup(operand.get())) {
        operand.set(copiedValue);
        continue;
      }

      // Check that we can actually copy this definition inside.
      if (!definingOp->hasTrait<ConstantLike>()) {
        auto d = op->emitError("operation in clock tree uses external value");
        d.attachNote() << "clock trees can only use external constant values";
        d.attachNote(definingOp->getLoc()) << "external value defined here:";
        d.attachNote(clockOp->getLoc()) << "clock tree:";
        return WalkResult::interrupt();
      }

      // Copy the op inside the clock tree (or move it if all uses are within
      // the clock tree).
      bool canMove = llvm::all_of(definingOp->getUsers(), [&](Operation *user) {
        return clockRegion->isAncestor(user->getParentRegion());
      });
      Operation *clonedOp;
      if (canMove) {
        definingOp->remove();
        clonedOp = definingOp;
        ++numOpsMoved;
      } else {
        clonedOp = definingOp->cloneWithoutRegions();
        ++numOpsCopied;
      }
      builder.insert(clonedOp);
      if (!canMove) {
        for (auto [outerResult, innerResult] :
             llvm::zip(definingOp->getResults(), clonedOp->getResults())) {
          copiedValues.insert({outerResult, innerResult});
          if (operand.get() == outerResult)
            operand.set(innerResult);
        }
      }
    }
    return WalkResult::advance();
  });
  return success(!result.wasInterrupted());
}

std::unique_ptr<Pass> arc::createLowerClocksToFuncsPass() {
  return std::make_unique<LowerClocksToFuncsPass>();
}
