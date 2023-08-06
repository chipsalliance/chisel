//===- ArcFolds.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

using namespace circt;
using namespace arc;
using namespace mlir;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static bool isAlways(Attribute attr, bool expected) {
  if (auto enable = dyn_cast_or_null<IntegerAttr>(attr))
    return enable.getValue().getBoolValue() == expected;
  return false;
}

static bool isAlways(Value value, bool expected) {
  if (!value)
    return false;

  if (auto constOp = value.getDefiningOp<hw::ConstantOp>())
    return constOp.getValue().getBoolValue() == expected;

  return false;
}

//===----------------------------------------------------------------------===//
// StateOp
//===----------------------------------------------------------------------===//

LogicalResult StateOp::fold(FoldAdaptor adaptor,
                            SmallVectorImpl<OpFoldResult> &results) {
  if ((isAlways(adaptor.getEnable(), false) ||
       isAlways(adaptor.getReset(), true)) &&
      !getOperation()->hasAttr("name") && !getOperation()->hasAttr("names")) {
    // We can fold to zero here because the states are zero-initialized and
    // don't ever change.
    for (auto resTy : getResultTypes())
      results.push_back(IntegerAttr::get(resTy, 0));
    return success();
  }

  // Remove operand when input is default value.
  if (isAlways(adaptor.getReset(), false))
    return getResetMutable().clear(), success();

  // Remove operand when input is default value.
  if (isAlways(adaptor.getEnable(), true))
    return getEnableMutable().clear(), success();

  return failure();
}

LogicalResult StateOp::canonicalize(StateOp op, PatternRewriter &rewriter) {
  // When there are no names attached, the state is not externaly observable.
  // When there are also no internal users, we can remove it.
  if (op->use_empty() && !op->hasAttr("name") && !op->hasAttr("names")) {
    rewriter.eraseOp(op);
    return success();
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// MemoryWriteOp
//===----------------------------------------------------------------------===//

LogicalResult MemoryWriteOp::fold(FoldAdaptor adaptor,
                                  SmallVectorImpl<OpFoldResult> &results) {
  if (isAlways(adaptor.getEnable(), true))
    return getEnableMutable().clear(), success();
  return failure();
}

LogicalResult MemoryWriteOp::canonicalize(MemoryWriteOp op,
                                          PatternRewriter &rewriter) {
  if (isAlways(op.getEnable(), false))
    return rewriter.eraseOp(op), success();
  return failure();
}

//===----------------------------------------------------------------------===//
// StorageGetOp
//===----------------------------------------------------------------------===//

LogicalResult StorageGetOp::canonicalize(StorageGetOp op,
                                         PatternRewriter &rewriter) {
  if (auto pred = op.getStorage().getDefiningOp<StorageGetOp>()) {
    op.getStorageMutable().assign(pred.getStorage());
    op.setOffset(op.getOffset() + pred.getOffset());
    return success();
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// ClockDomainOp
//===----------------------------------------------------------------------===//

static bool removeUnusedClockDomainInputs(ClockDomainOp op,
                                          PatternRewriter &rewriter) {
  BitVector toDelete(op.getBodyBlock().getNumArguments());
  for (auto arg : llvm::reverse(op.getBodyBlock().getArguments())) {
    if (arg.use_empty()) {
      auto i = arg.getArgNumber();
      toDelete.set(i);
      op.getInputsMutable().erase(i);
    }
  }
  op.getBodyBlock().eraseArguments(toDelete);
  return toDelete.any();
}

static bool removeUnusedClockDomainOutputs(ClockDomainOp op,
                                           PatternRewriter &rewriter) {
  SmallVector<Type> resultTypes;
  for (auto res : llvm::reverse(op->getResults())) {
    if (res.use_empty())
      op.getBodyBlock().getTerminator()->eraseOperand(res.getResultNumber());
    else
      resultTypes.push_back(res.getType());
  }

  // Nothing is changed.
  if (resultTypes.size() == op->getNumResults())
    return false;

  rewriter.setInsertionPoint(op);

  auto newDomain = rewriter.create<ClockDomainOp>(
      op.getLoc(), resultTypes, op.getInputs(), op.getClock());
  rewriter.inlineRegionBefore(op.getBody(), newDomain.getBody(),
                              newDomain->getRegion(0).begin());

  unsigned currIdx = 0;
  for (auto result : op.getOutputs()) {
    if (!result.use_empty())
      rewriter.replaceAllUsesWith(result, newDomain->getResult(currIdx++));
  }

  rewriter.eraseOp(op);
  return true;
}

LogicalResult ClockDomainOp::canonicalize(ClockDomainOp op,
                                          PatternRewriter &rewriter) {
  rewriter.setInsertionPointToStart(&op.getBodyBlock());

  // Canonicalize inputs
  DenseMap<Value, unsigned> seenArgs;
  for (auto arg :
       llvm::make_early_inc_range(op.getBodyBlock().getArguments())) {
    auto i = arg.getArgNumber();
    auto inputVal = op.getInputs()[i];

    if (arg.use_empty())
      continue;

    // Remove duplicate inputs
    if (seenArgs.count(inputVal)) {
      rewriter.replaceAllUsesWith(
          arg, op.getBodyBlock().getArgument(seenArgs[inputVal]));
      continue;
    }

    // Pull in memories that are only used in this clock domain and clone
    // constants into the clock domain.
    if (auto *inputOp = inputVal.getDefiningOp()) {
      bool isConstant = inputOp->hasTrait<OpTrait::ConstantLike>();
      bool hasOneUse = inputVal.hasOneUse();
      if (isConstant || (isa<MemoryOp>(inputOp) && hasOneUse)) {
        auto resultNumber = cast<OpResult>(inputVal).getResultNumber();
        auto *clone = rewriter.clone(*inputOp);
        rewriter.replaceAllUsesWith(arg, clone->getResult(resultNumber));
        if (hasOneUse && inputOp->getNumResults() == 1) {
          inputVal.dropAllUses();
          rewriter.eraseOp(inputOp);
        }
        continue;
      }
    }

    seenArgs[op.getInputs()[i]] = i;
  }

  auto didCanonicalizeInput = removeUnusedClockDomainInputs(op, rewriter);

  // Canonicalize outputs
  for (auto [result, terminatorOperand] : llvm::zip(
           op.getOutputs(), op.getBodyBlock().getTerminator()->getOperands())) {
    // Replace results which are just passed-through inputs with the input
    // directly. This makes this result unused and is thus removed later on.
    if (isa<BlockArgument>(terminatorOperand))
      rewriter.replaceAllUsesWith(
          result, op.getInputs()[cast<BlockArgument>(terminatorOperand)
                                     .getArgNumber()]);

    // Outputs that are just constant operations can be replaced by a clone of
    // the constant outside of the clock domain. This makes the result unused
    // and is thus removed later on.
    // TODO: we could also push out all operations that are not clocked/don't
    // have side-effects. If there are long chains of such operations this can
    // lead to long canonicalizer runtimes though, so we need to be careful here
    // and maybe do it as a separate pass (or make sure that such chains are
    // never pulled into the clock domain in the first place).
    if (auto *defOp = terminatorOperand.getDefiningOp();
        defOp && defOp->hasTrait<OpTrait::ConstantLike>() &&
        !result.use_empty()) {
      rewriter.setInsertionPointAfter(op);
      unsigned resultIdx = cast<OpResult>(terminatorOperand).getResultNumber();
      auto *clone = rewriter.clone(*defOp);
      if (defOp->hasOneUse()) {
        defOp->dropAllUses();
        rewriter.eraseOp(defOp);
      }
      rewriter.replaceAllUsesWith(result, clone->getResult(resultIdx));
    }
  }

  auto didCanoncalizeOutput = removeUnusedClockDomainOutputs(op, rewriter);

  return success(didCanonicalizeInput || didCanoncalizeOutput);
}
