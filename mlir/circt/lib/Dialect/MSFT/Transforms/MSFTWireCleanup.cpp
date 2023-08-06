//===- MSFTWireCleanup.cpp - Wire cleanup pass ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MSFTPassCommon.h"
#include "PassDetails.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/MSFT/MSFTDialect.h"
#include "circt/Dialect/MSFT/MSFTOpInterfaces.h"
#include "circt/Dialect/MSFT/MSFTOps.h"
#include "circt/Dialect/MSFT/MSFTPasses.h"
#include "circt/Support/Namespace.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace circt;
using namespace msft;

namespace {
struct WireCleanupPass : public WireCleanupBase<WireCleanupPass>,
                         MSFTPassCommon {
  void runOnOperation() override;
};
} // anonymous namespace

void WireCleanupPass::runOnOperation() {
  ModuleOp topMod = getOperation();
  topLevelSyms.addDefinitions(topMod);
  if (failed(verifyInstances(topMod))) {
    signalPassFailure();
    return;
  }

  SmallVector<MSFTModuleOp> sortedMods;
  getAndSortModules(topMod, sortedMods);

  for (auto mod : sortedMods) {
    bubbleWiresUp(mod);
    dedupOutputs(mod);
  }

  for (auto mod : llvm::reverse(sortedMods)) {
    sinkWiresDown(mod);
    dedupInputs(mod);
  }
}

/// Remove outputs driven by the same value.
void MSFTPassCommon::dedupOutputs(MSFTModuleOp mod) {
  Block *body = mod.getBodyBlock();
  Operation *terminator = body->getTerminator();

  DenseMap<Value, unsigned> valueToOutputIdx;
  SmallVector<unsigned> outputMap;
  llvm::BitVector outputPortsToRemove(terminator->getNumOperands());
  for (OpOperand &outputVal : terminator->getOpOperands()) {
    auto existing = valueToOutputIdx.find(outputVal.get());
    if (existing != valueToOutputIdx.end()) {
      outputMap.push_back(existing->second);
      outputPortsToRemove.set(outputVal.getOperandNumber());
    } else {
      outputMap.push_back(valueToOutputIdx.size());
      valueToOutputIdx[outputVal.get()] = valueToOutputIdx.size();
    }
  }

  mod.removePorts(llvm::BitVector(mod.getNumArguments()), outputPortsToRemove);
  updateInstances(mod, makeSequentialRange(mod.getNumResults()),
                  [&](InstanceOp newInst, InstanceOp oldInst,
                      SmallVectorImpl<Value> &newOperands) {
                    // Operands don't change.
                    llvm::append_range(newOperands, oldInst.getOperands());
                    // The results have to be remapped.
                    for (OpResult res : oldInst.getResults())
                      res.replaceAllUsesWith(
                          newInst.getResult(outputMap[res.getResultNumber()]));
                  });
}

/// Push up any wires which are simply passed-through.
void MSFTPassCommon::bubbleWiresUp(MSFTModuleOp mod) {
  Block *body = mod.getBodyBlock();
  Operation *terminator = body->getTerminator();
  hw::ModulePortInfo ports = mod.getPortList();

  // Find all "passthough" internal wires, filling 'inputPortsToRemove' as a
  // side-effect.
  DenseMap<Value, hw::PortInfo> passThroughs;
  llvm::BitVector inputPortsToRemove(ports.sizeInputs());
  for (hw::PortInfo inputPort : ports.getInputs()) {
    BlockArgument portArg = body->getArgument(inputPort.argNum);
    bool removePort = true;
    for (OpOperand user : portArg.getUsers()) {
      if (user.getOwner() == terminator)
        passThroughs[portArg] = inputPort;
      else
        removePort = false;
    }
    if (removePort)
      inputPortsToRemove.set(inputPort.argNum);
  }

  // Find all output ports which we can remove. Fill in 'outputToInputIdx' to
  // help rewire instantiations later on.
  DenseMap<unsigned, unsigned> outputToInputIdx;
  llvm::BitVector outputPortsToRemove(ports.sizeOutputs());
  for (hw::PortInfo outputPort : ports.getOutputs()) {
    assert(outputPort.argNum < terminator->getNumOperands() && "Invalid IR");
    Value outputValue = terminator->getOperand(outputPort.argNum);
    auto inputNumF = passThroughs.find(outputValue);
    if (inputNumF == passThroughs.end())
      continue;
    hw::PortInfo inputPort = inputNumF->second;
    outputToInputIdx[outputPort.argNum] = inputPort.argNum;
    outputPortsToRemove.set(outputPort.argNum);
  }

  // Use MSFTModuleOp's `removePorts` method to remove the ports. It returns a
  // mapping of the new output port to old output port indices to assist in
  // updating the instantiations later on.
  auto newToOldResult =
      mod.removePorts(inputPortsToRemove, outputPortsToRemove);

  // Update the instantiations.
  auto setPassthroughsGetOperands = [&](InstanceOp newInst, InstanceOp oldInst,
                                        SmallVectorImpl<Value> &newOperands) {
    // Re-map the passthrough values around the instance.
    for (auto idxPair : outputToInputIdx) {
      size_t outputPortNum = idxPair.first;
      assert(outputPortNum <= oldInst.getNumResults());
      size_t inputPortNum = idxPair.second;
      assert(inputPortNum <= oldInst.getNumOperands());
      oldInst.getResult(outputPortNum)
          .replaceAllUsesWith(oldInst.getOperand(inputPortNum));
    }
    // Use a sort-merge-join approach to figure out the operand mapping on the
    // fly.
    for (size_t operNum = 0, e = oldInst.getNumOperands(); operNum < e;
         ++operNum)
      if (!inputPortsToRemove.test(operNum))
        newOperands.push_back(oldInst.getOperand(operNum));
  };
  updateInstances(mod, newToOldResult, setPassthroughsGetOperands);
}

std::unique_ptr<Pass> circt::msft::createWireCleanupPass() {
  return std::make_unique<WireCleanupPass>();
}
