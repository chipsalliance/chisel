//===- MakeTables.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arc-lookup-tables"

using namespace circt;
using namespace arc;
using namespace hw;

namespace {

static constexpr int tableMinOpCount = 20;
static constexpr int tableMaxSize = 32768; // bits

struct MakeTablesPass : public MakeTablesBase<MakeTablesPass> {
  void runOnOperation() override;
  void runOnArc(DefineOp defineOp);
};
} // namespace

static inline uint32_t bitsMask(uint32_t nbits) {
  if (nbits == 32)
    return ~0;
  return (1 << nbits) - 1;
}

static inline uint32_t bitsGet(uint32_t x, uint32_t lb, uint32_t ub) {
  return (x >> lb) & bitsMask(ub - lb + 1);
}

void MakeTablesPass::runOnOperation() {
  auto module = getOperation();
  for (auto op : module.getOps<DefineOp>())
    runOnArc(op);
}

void MakeTablesPass::runOnArc(DefineOp defineOp) {
  // Determine the number of input bits.
  unsigned numInputBits = 0;
  for (auto &type : defineOp.getArgumentTypes()) {
    auto intType = type.dyn_cast<IntegerType>();
    if (!intType)
      return;
    numInputBits += intType.getWidth();
  }
  if (numInputBits == 0)
    return;

  // Count the number of non-constant operations in the block.
  unsigned numOps = 0;
  for (auto &op : defineOp.getBodyBlock().without_terminator())
    if (!op.hasTrait<OpTrait::ConstantLike>())
      ++numOps;

  // Determine the number of output bits.
  unsigned numOutputBits = 0;
  auto outputOp = cast<arc::OutputOp>(defineOp.getBodyBlock().getTerminator());
  for (auto type : outputOp.getOperandTypes()) {
    auto intType = type.dyn_cast<IntegerType>();
    if (!intType)
      return;
    numOutputBits += intType.getWidth();
  }
  if (numOutputBits == 0)
    return;

  LLVM_DEBUG(llvm::dbgs() << "Making lookup tables in `" << defineOp.getName()
                          << "`\n");
  LLVM_DEBUG(llvm::dbgs() << "- " << numInputBits << " input bits, "
                          << numOutputBits << " output bits, " << numOps
                          << " ops\n");

  // Check whether the table dimensions are within bounds.
  if (numInputBits >= 31) {
    LLVM_DEBUG(llvm::dbgs() << "- Skip; too many input bits\n");
    return;
  }
  if (numOps < tableMinOpCount) {
    LLVM_DEBUG(llvm::dbgs() << "- Skip; not enough ops\n");
    return;
  }

  unsigned numTableEntries = 1U << numInputBits;
  if (numTableEntries > tableMaxSize / numOutputBits) {
    LLVM_DEBUG(llvm::dbgs() << "- Skip; table too large\n");
    return;
  }
  LLVM_DEBUG(llvm::dbgs() << "- Creating table of "
                          << numTableEntries * numOutputBits << " bits\n");

  // Actually build the table.
  SmallVector<Operation *, 64> tabularizedOps;
  for (auto &op : defineOp.getBodyBlock().without_terminator())
    tabularizedOps.push_back(&op);

  // Concatenate the inputs into a single index value.
  auto builder = ImplicitLocOpBuilder::atBlockBegin(defineOp.getLoc(),
                                                    &defineOp.getBodyBlock());
  SmallVector<Value> inputsToConcat(defineOp.getArguments());
  std::reverse(inputsToConcat.begin(), inputsToConcat.end());
  auto concatInputs = inputsToConcat.size() > 1
                          ? builder.create<comb::ConcatOp>(inputsToConcat)
                          : inputsToConcat[0];

  // Compute a lookup table for every output.
  SmallVector<SmallVector<Attribute, 0>> tables;
  DenseMap<Value, Attribute> values;
  tables.resize(outputOp->getNumOperands());

  for (int input = (1U << numInputBits) - 1; input >= 0; input--) {
    // Assign the input values.
    values.clear();
    unsigned bits = 0;
    for (auto arg : defineOp.getArguments()) {
      auto w = arg.getType().dyn_cast<IntegerType>().getWidth();
      values[arg] = builder.getIntegerAttr(arg.getType(),
                                           bitsGet(input, bits, bits + w - 1));
      bits += w;
    }

    // Evaluate the operations.
    SmallVector<Attribute> constants;
    for (auto *operation : tabularizedOps) {
      constants.clear();
      for (auto operand : operation->getOperands())
        constants.push_back(values[operand]);

      SmallVector<OpFoldResult, 8> resultValues;
      if (failed(operation->fold(constants, resultValues))) {
        LLVM_DEBUG(llvm::dbgs() << "- Skip; operation folder failed\n");
        return;
      }

      for (auto [result, resultValue] :
           llvm::zip(operation->getResults(), resultValues)) {
        auto attr = resultValue.dyn_cast<Attribute>();
        if (!attr)
          attr = values[resultValue.dyn_cast<Value>()];
        values[result] = attr;
      }
    }

    // Add the evaluated values to the output tables.
    for (auto [table, outputOperand] :
         llvm::zip(tables, outputOp->getOpOperands())) {
      table.push_back(values[outputOperand.get()].dyn_cast<Attribute>());
    }
  }

  // Create the table lookup ops.
  for (auto [table, outputOperand] :
       llvm::zip(tables, outputOp->getOpOperands())) {
    auto array = builder.create<hw::AggregateConstantOp>(
        ArrayType::get(outputOperand.get().getType(), numTableEntries),
        builder.getArrayAttr(table));
    outputOperand.set(builder.create<hw::ArrayGetOp>(array, concatInputs));
  }

  for (auto *op : tabularizedOps) {
    op->dropAllUses();
    op->erase();
  }
}

std::unique_ptr<Pass> arc::createMakeTablesPass() {
  return std::make_unique<MakeTablesPass>();
}
