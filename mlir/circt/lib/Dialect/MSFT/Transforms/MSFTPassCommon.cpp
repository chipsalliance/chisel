//===- MSFTPassCommon.cpp - MSFT Pass Common --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MSFTPassCommon.h"
#include "circt/Dialect/MSFT/MSFTPasses.h"

using namespace mlir;
using namespace circt;
using namespace msft;

void MSFTPassCommon::dedupInputs(MSFTModuleOp mod) {
  const auto &instantiations = moduleInstantiations[mod];
  // TODO: remove this limitation. This would involve looking at the common
  // loopbacks for all the instances.
  if (instantiations.size() != 1)
    return;
  InstanceOp inst =
      dyn_cast<InstanceOp>(static_cast<Operation *>(instantiations[0]));
  if (!inst)
    return;

  // Find all the arguments which are driven by the same signal. Remap them
  // appropriately within the module, and mark that input port for deletion.
  Block *body = mod.getBodyBlock();
  DenseMap<Value, unsigned> valueToInput;
  llvm::BitVector argsToErase(body->getNumArguments());
  for (OpOperand &oper : inst->getOpOperands()) {
    auto existingValue = valueToInput.find(oper.get());
    if (existingValue != valueToInput.end()) {
      unsigned operNum = oper.getOperandNumber();
      unsigned duplicateInputNum = existingValue->second;
      body->getArgument(operNum).replaceAllUsesWith(
          body->getArgument(duplicateInputNum));
      argsToErase.set(operNum);
    } else {
      valueToInput[oper.get()] = oper.getOperandNumber();
    }
  }

  // Remove the ports.
  auto remappedResults =
      mod.removePorts(argsToErase, llvm::BitVector(inst.getNumResults()));
  // and update the instantiations.
  auto getOperands = [&](InstanceOp newInst, InstanceOp oldInst,
                         SmallVectorImpl<Value> &newOperands) {
    for (unsigned argNum = 0, e = oldInst.getNumOperands(); argNum < e;
         ++argNum)
      if (!argsToErase.test(argNum))
        newOperands.push_back(oldInst.getOperand(argNum));
  };
  inst = updateInstances(mod, remappedResults, getOperands)[0];

  SmallVector<Attribute, 32> newArgNames;
  std::string buff;
  for (Value oper : inst->getOperands()) {
    newArgNames.push_back(StringAttr::get(
        mod.getContext(), getValueName(oper, topLevelSyms, buff)));
  }
  mod.setArgNamesAttr(ArrayAttr::get(mod.getContext(), newArgNames));
}

/// Sink all the instance connections which are loops.
void MSFTPassCommon::sinkWiresDown(MSFTModuleOp mod) {
  const auto &instantiations = moduleInstantiations[mod];
  // TODO: remove this limitation. This would involve looking at the common
  // loopbacks for all the instances.
  if (instantiations.size() != 1)
    return;
  InstanceOp inst =
      dyn_cast<InstanceOp>(static_cast<Operation *>(instantiations[0]));
  if (!inst)
    return;

  // Find all the "loopback" connections in the instantiation. Populate
  // 'inputToOutputLoopback' with a mapping of input port to output port which
  // drives it. Populate 'resultsToErase' with output ports which only drive
  // input ports.
  DenseMap<unsigned, unsigned> inputToOutputLoopback;
  llvm::BitVector resultsToErase(inst.getNumResults());
  for (unsigned resNum = 0, e = inst.getNumResults(); resNum < e; ++resNum) {
    bool allLoops = true;
    for (auto &use : inst.getResult(resNum).getUses()) {
      if (use.getOwner() != inst.getOperation())
        allLoops = false;
      else
        inputToOutputLoopback[use.getOperandNumber()] = resNum;
    }
    if (allLoops)
      resultsToErase.set(resNum);
  }

  // Add internal connections to replace the instantiation's loop back
  // connections.
  Block *body = mod.getBodyBlock();
  Operation *terminator = body->getTerminator();
  llvm::BitVector argsToErase(body->getNumArguments());
  for (auto resOper : inputToOutputLoopback) {
    body->getArgument(resOper.first)
        .replaceAllUsesWith(terminator->getOperand(resOper.second));
    argsToErase.set(resOper.first);
  }

  // Remove the ports.
  SmallVector<unsigned> newToOldResultMap =
      mod.removePorts(argsToErase, resultsToErase);
  // and update the instantiations.
  auto getOperands = [&](InstanceOp newInst, InstanceOp oldInst,
                         SmallVectorImpl<Value> &newOperands) {
    // Use sort-merge-join to compute the new operands;
    for (unsigned argNum = 0, e = oldInst.getNumOperands(); argNum < e;
         ++argNum)
      if (!argsToErase.test(argNum))
        newOperands.push_back(oldInst.getOperand(argNum));
  };
  updateInstances(mod, newToOldResultMap, getOperands);
}

void MSFTPassCommon::getAndSortModules(ModuleOp topMod,
                                       SmallVectorImpl<MSFTModuleOp> &mods) {
  SmallVector<hw::HWModuleLike, 16> moduleLikes;
  PassCommon::getAndSortModules(topMod, moduleLikes);
  mods.clear();
  for (auto modLike : moduleLikes) {
    auto mod = dyn_cast<MSFTModuleOp>(modLike.getOperation());
    if (mod)
      mods.push_back(mod);
  }
}

SmallVector<InstanceOp, 1> MSFTPassCommon::updateInstances(
    MSFTModuleOp mod, ArrayRef<unsigned> newToOldResultMap,
    llvm::function_ref<void(InstanceOp, InstanceOp, SmallVectorImpl<Value> &)>
        getOperandsFunc) {

  SmallVector<hw::HWInstanceLike, 1> newInstances;
  SmallVector<InstanceOp, 1> newMsftInstances;
  for (hw::HWInstanceLike instLike : moduleInstantiations[mod]) {
    assert(instLike->getParentOp());
    auto inst = dyn_cast<InstanceOp>(instLike.getOperation());
    if (!inst) {
      instLike.emitWarning("Can not update hw.instance ops");
      continue;
    }

    OpBuilder b(inst);
    auto newInst = b.create<InstanceOp>(inst.getLoc(), mod.getResultTypes(),
                                        inst.getOperands(), inst->getAttrs());

    SmallVector<Value> newOperands;
    getOperandsFunc(newInst, inst, newOperands);
    newInst->setOperands(newOperands);

    for (auto oldResult : llvm::enumerate(newToOldResultMap))
      if (oldResult.value() < inst.getNumResults())
        inst.getResult(oldResult.value())
            .replaceAllUsesWith(newInst.getResult(oldResult.index()));

    newInstances.push_back(newInst);
    newMsftInstances.push_back(newInst);
    inst->dropAllUses();
    inst->erase();
  }
  moduleInstantiations[mod].swap(newInstances);
  return newMsftInstances;
}
