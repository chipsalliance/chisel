//===- SplitLoops.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Support/Namespace.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arc-split-loops"

using namespace circt;
using namespace arc;
using namespace hw;

using llvm::SmallSetVector;

//===----------------------------------------------------------------------===//
// Arc Splitter
//===----------------------------------------------------------------------===//

namespace {
/// A value imported into a split.
struct ImportedValue {
  /// Indicates where this value originates from. If true, the value is an input
  /// of the original arc. If false the value is exported from a different
  /// split.
  unsigned isInput : 1;
  /// The original arc's input number, or the result number of the split that
  /// exports this value.
  unsigned index : 15;
  /// If this value is exported from a different split, this is that split's
  /// index.
  unsigned split : 16;
};

/// A single arc split out from another arc.
struct Split {
  Split(MLIRContext *context, unsigned index, const APInt &color)
      : index(index), color(color), block(std::make_unique<Block>()),
        builder(context) {
    builder.setInsertionPointToStart(block.get());
  }

  /// Map an input of the original arc into this split.
  void importInput(BlockArgument arg) {
    importedValues.push_back({true, arg.getArgNumber(), 0});
    mapping.map(arg, block->addArgument(arg.getType(), arg.getLoc()));
  }

  /// Map a value in a different split into this split.
  void importFromOtherSplit(Value value, Split &otherSplit) {
    auto resultIdx = otherSplit.exportValue(value);
    importedValues.push_back({false, resultIdx, otherSplit.index});
    mapping.map(value, block->addArgument(value.getType(), value.getLoc()));
  }

  /// Export a value in this split as an output. Returns result number this
  /// value will have.
  unsigned exportValue(Value value) {
    value = mapping.lookup(value);
    auto result = exportedValueIndices.insert({value, exportedValues.size()});
    if (result.second)
      exportedValues.push_back(value);
    return result.first->second;
  }

  unsigned index;
  APInt color;

  std::unique_ptr<Block> block;
  OpBuilder builder;
  IRMapping mapping;

  /// Where each value mapped to a block argument is coming from.
  SmallVector<ImportedValue> importedValues;
  /// Which values of this split are exposed as outputs.
  SmallVector<Value> exportedValues;
  SmallDenseMap<Value, unsigned> exportedValueIndices;
};

/// Helper structure to split one arc into multiple ones.
struct Splitter {
  Splitter(MLIRContext *context, Location loc) : context(context), loc(loc) {}
  void run(Block &block, DenseMap<Operation *, APInt> &coloring);
  Split &getSplit(const APInt &color);

  MLIRContext *context;
  Location loc;

  /// A split for each distinct operation coloring in the original arc.
  SmallVector<Split *> splits;
  SmallDenseMap<APInt, std::unique_ptr<Split>> splitsByColor;

  /// Where each of the original arc's outputs come from after splitting.
  SmallVector<ImportedValue> outputs;
};
} // namespace

/// Create a separate arc body block for every unique coloring of operations.
void Splitter::run(Block &block, DenseMap<Operation *, APInt> &coloring) {
  for (auto &op : block.without_terminator()) {
    auto color = coloring.lookup(&op);
    auto &split = getSplit(color);

    // Collect the operands of the current operation.
    SmallSetVector<Value, 4> operands;
    op.walk([&](Operation *op) {
      for (auto operand : op->getOperands())
        if (operand.getParentBlock() == &block)
          operands.insert(operand);
    });

    // Each operand that is either an input of the original arc or that is
    // defined by an operation that got moved to a different split, create an
    // input to the current split for that value.
    for (auto operand : operands) {
      if (split.mapping.contains(operand))
        continue;
      if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
        split.importInput(blockArg);
        continue;
      }
      auto *operandOp = operand.getDefiningOp();
      auto operandColor = coloring.lookup(operandOp);
      assert(operandOp && color != operandColor);
      auto &operandSplit = getSplit(operandColor);
      split.importFromOtherSplit(operand, operandSplit);
    }

    // Move the operation into the split.
    split.builder.clone(op, split.mapping);
  }

  // Reconstruct where each of the original arc outputs got mapped to.
  for (auto operand : block.getTerminator()->getOperands()) {
    if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
      outputs.push_back({true, blockArg.getArgNumber(), 0});
      continue;
    }
    auto &operandSplit = getSplit(coloring.lookup(operand.getDefiningOp()));
    auto resultIdx = operandSplit.exportValue(operand);
    outputs.push_back({false, resultIdx, operandSplit.index});
  }

  // Create the final `arc.output` op for each of the splits.
  for (auto &split : splits)
    split->builder.create<arc::OutputOp>(loc, split->exportedValues);
}

/// Get or create the split for a given operation color.
Split &Splitter::getSplit(const APInt &color) {
  auto &split = splitsByColor[color];
  if (!split) {
    auto index = splits.size();
    LLVM_DEBUG(llvm::dbgs()
               << "- Creating split " << index << " for " << color << "\n");
    split = std::make_unique<Split>(context, index, color);
    splits.push_back(split.get());
  }
  return *split;
}

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

namespace {
struct SplitLoopsPass : public SplitLoopsBase<SplitLoopsPass> {
  void runOnOperation() override;
  void splitArc(Namespace &arcNamespace, DefineOp defOp,
                ArrayRef<StateOp> arcUses);
  void replaceArcUse(StateOp arcUse, ArrayRef<DefineOp> splitDefs,
                     ArrayRef<Split *> splits, ArrayRef<ImportedValue> outputs);
  LogicalResult ensureNoLoops();

  DenseSet<StateOp> allArcUses;
};
} // namespace

void SplitLoopsPass::runOnOperation() {
  auto module = getOperation();
  allArcUses.clear();

  // Collect all arc definitions.
  Namespace arcNamespace;
  DenseMap<StringAttr, DefineOp> arcDefs;
  for (auto arcDef : module.getOps<DefineOp>()) {
    arcNamespace.newName(arcDef.getSymName());
    arcDefs[arcDef.getSymNameAttr()] = arcDef;
  }

  // Collect all arc uses and determine which arcs we should split.
  SetVector<DefineOp> arcsToSplit;
  DenseMap<DefineOp, SmallVector<StateOp>> arcUses;
  SetVector<StateOp> allArcUses;

  module.walk([&](StateOp stateOp) {
    auto sym = stateOp.getArcAttr().getAttr();
    auto defOp = arcDefs.lookup(sym);
    arcUses[defOp].push_back(stateOp);
    allArcUses.insert(stateOp);
    if (stateOp.getLatency() == 0 && stateOp.getNumResults() > 1)
      arcsToSplit.insert(defOp);
  });

  // Split all arcs with more than one result.
  // TODO: This is ugly and we should only split arcs that are truly involved in
  // a loop. But detecting the minimal split among the arcs is fairly
  // non-trivial and needs a dedicated implementation effort.
  for (auto defOp : arcsToSplit)
    splitArc(arcNamespace, defOp, arcUses[defOp]);

  // Ensure that there are no loops through arcs remaining.
  if (failed(ensureNoLoops()))
    return signalPassFailure();
}

/// Split a single arc into a separate arc for each result.
void SplitLoopsPass::splitArc(Namespace &arcNamespace, DefineOp defOp,
                              ArrayRef<StateOp> arcUses) {
  LLVM_DEBUG(llvm::dbgs() << "Splitting arc " << defOp.getSymNameAttr()
                          << "\n");

  // Mark the operations in the arc according to which result they contribute
  // to.
  auto numResults = defOp.getNumResults();
  DenseMap<Value, APInt> valueColoring;
  DenseMap<Operation *, APInt> opColoring;

  for (auto &operand : defOp.getBodyBlock().getTerminator()->getOpOperands())
    valueColoring.insert(
        {operand.get(),
         APInt::getOneBitSet(numResults, operand.getOperandNumber())});

  for (auto &op : llvm::reverse(defOp.getBodyBlock().without_terminator())) {
    auto coloring = APInt::getZero(numResults);
    for (auto result : op.getResults())
      if (auto it = valueColoring.find(result); it != valueColoring.end())
        coloring |= it->second;
    opColoring.insert({&op, coloring});
    op.walk([&](Operation *op) {
      for (auto &operand : op->getOpOperands())
        valueColoring.try_emplace(operand.get(), numResults, 0).first->second |=
            coloring;
    });
  }

  // Determine the splits for this arc.
  Splitter splitter(&getContext(), defOp.getLoc());
  splitter.run(defOp.getBodyBlock(), opColoring);

  // Materialize the split arc definitions.
  ImplicitLocOpBuilder builder(defOp.getLoc(), defOp);
  SmallVector<DefineOp> splitArcs;
  splitArcs.reserve(splitter.splits.size());
  for (auto &split : splitter.splits) {
    auto splitName = defOp.getSymName();
    if (splitter.splits.size() > 1)
      splitName = arcNamespace.newName(defOp.getSymName() + "_split_" +
                                       Twine(split->index));
    auto splitArc = builder.create<DefineOp>(
        splitName, builder.getFunctionType(
                       split->block->getArgumentTypes(),
                       split->block->getTerminator()->getOperandTypes()));
    splitArc.getBody().push_back(split->block.release());
    splitArcs.push_back(splitArc);
  }

  // Replace all uses with the new splits and remove the old definition.
  for (auto arcUse : arcUses)
    replaceArcUse(arcUse, splitArcs, splitter.splits, splitter.outputs);
  defOp.erase();
}

/// Replace a use of the original arc with new uses for the splits.
void SplitLoopsPass::replaceArcUse(StateOp arcUse, ArrayRef<DefineOp> splitDefs,
                                   ArrayRef<Split *> splits,
                                   ArrayRef<ImportedValue> outputs) {
  ImplicitLocOpBuilder builder(arcUse.getLoc(), arcUse);
  SmallVector<StateOp> newUses(splits.size());

  // Resolve an `ImportedValue` to either an operand of the original arc or the
  // result of another split.
  auto getMappedValue = [&](ImportedValue value) {
    if (value.isInput)
      return arcUse.getInputs()[value.index];
    return newUses[value.split].getResult(value.index);
  };

  // Collect the operands for each split and create a new use for each. These
  // are either operands of the original arc, or values from other splits
  // exported as results.
  DenseMap<unsigned, unsigned> splitIdxMap;
  for (auto [i, split] : llvm::enumerate(splits))
    splitIdxMap[split->index] = i;

  DenseSet<unsigned> splitsDone;
  SmallVector<std::pair<const DefineOp, const Split *>> worklist;

  auto getMappedValuesOrSchedule = [&](ArrayRef<ImportedValue> importedValues,
                                       SmallVector<Value> &operands) {
    for (auto importedValue : importedValues) {
      if (!importedValue.isInput && !splitsDone.contains(importedValue.split)) {
        unsigned idx = splitIdxMap[importedValue.split];
        worklist.push_back({splitDefs[idx], splits[idx]});
        return false;
      }

      operands.push_back(getMappedValue(importedValue));
    }

    return true;
  };

  // Initialize worklist
  for (auto [splitDef, split] : llvm::reverse(llvm::zip(splitDefs, splits)))
    worklist.push_back({splitDef, split});

  // Process worklist
  while (!worklist.empty()) {
    auto [splitDef, split] = worklist.back();

    if (splitsDone.contains(split->index)) {
      worklist.pop_back();
      continue;
    }

    SmallVector<Value> operands;
    if (!getMappedValuesOrSchedule(split->importedValues, operands))
      continue;

    auto newUse =
        builder.create<StateOp>(splitDef, Value{}, Value{}, 0, operands);
    allArcUses.insert(newUse);
    newUses[split->index] = newUse;

    splitsDone.insert(split->index);
    worklist.pop_back();
  }

  // Update the users of the original arc results.
  for (auto [result, importedValue] : llvm::zip(arcUse.getResults(), outputs))
    result.replaceAllUsesWith(getMappedValue(importedValue));
  allArcUses.erase(arcUse);
  arcUse.erase();
}

/// Check that there are no more zero-latency loops through arcs.
LogicalResult SplitLoopsPass::ensureNoLoops() {
  SmallVector<std::pair<Operation *, unsigned>, 0> worklist;
  DenseSet<Operation *> finished;
  DenseSet<Operation *> seen;
  for (auto op : allArcUses) {
    if (finished.contains(op))
      continue;
    assert(seen.empty());
    worklist.push_back({op, 0});
    while (!worklist.empty()) {
      auto [op, idx] = worklist.back();
      ++worklist.back().second;
      if (idx == op->getNumOperands()) {
        seen.erase(op);
        finished.insert(op);
        worklist.pop_back();
        continue;
      }
      auto operand = op->getOperand(idx);
      auto *def = operand.getDefiningOp();
      if (!def || finished.contains(def))
        continue;
      if (auto stateOp = dyn_cast<StateOp>(def);
          stateOp && stateOp.getLatency() > 0)
        continue;
      if (!seen.insert(def).second) {
        auto d = def->emitError(
            "loop splitting did not eliminate all loops; loop detected");
        for (auto [op, idx] : llvm::reverse(worklist)) {
          d.attachNote(op->getLoc())
              << "through operand " << (idx - 1) << " here:";
          if (op == def)
            break;
        }
        return failure();
      }
      worklist.push_back({def, 0});
    }
  }
  return success();
}

std::unique_ptr<Pass> arc::createSplitLoopsPass() {
  return std::make_unique<SplitLoopsPass>();
}
