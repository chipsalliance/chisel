//===- PortConverter.cpp - Module I/O rewriting utility ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/PortConverter.h"

#include <numeric>

using namespace circt;
using namespace hw;

/// Return a attribute with the specified suffix appended.
static StringAttr append(StringAttr base, const Twine &suffix) {
  if (suffix.isTriviallyEmpty())
    return base;
  auto *context = base.getContext();
  return StringAttr::get(context, base.getValue() + suffix);
}

namespace {

/// We consider non-caught ports to be ad-hoc signaling or 'untouched'. (Which
/// counts as a signaling protocol if one squints pretty hard). We mostly do
/// this since it allows us a more consistent internal API.
class UntouchedPortConversion : public PortConversion {
public:
  UntouchedPortConversion(PortConverterImpl &converter, hw::PortInfo origPort)
      : PortConversion(converter, origPort) {
    // Set the "RTTI flag" to true (see comment in header for this variable).
    isUntouchedFlag = true;
  }

  void mapInputSignals(OpBuilder &b, Operation *inst, Value instValue,
                       SmallVectorImpl<Value> &newOperands,
                       ArrayRef<Backedge> newResults) override {
    newOperands[portInfo.argNum] = instValue;
  }
  void mapOutputSignals(OpBuilder &b, Operation *inst, Value instValue,
                        SmallVectorImpl<Value> &newOperands,
                        ArrayRef<Backedge> newResults) override {
    instValue.replaceAllUsesWith(newResults[portInfo.argNum]);
  }

private:
  void buildInputSignals() override {
    Value newValue =
        converter.createNewInput(origPort, "", origPort.type, portInfo);
    if (body)
      body->getArgument(origPort.argNum).replaceAllUsesWith(newValue);
  }

  void buildOutputSignals() override {
    Value output;
    if (body)
      output = body->getTerminator()->getOperand(origPort.argNum);
    converter.createNewOutput(origPort, "", origPort.type, output, portInfo);
  }

  hw::PortInfo portInfo;
};

} // namespace

FailureOr<std::unique_ptr<PortConversion>>
PortConversionBuilder::build(hw::PortInfo port) {
  // Default builder is the 'untouched' port conversion which will simply
  // pass ports through unmodified.
  return {std::make_unique<UntouchedPortConversion>(converter, port)};
}

Value PortConverterImpl::createNewInput(PortInfo origPort, const Twine &suffix,
                                        Type type, PortInfo &newPort) {
  newPort = PortInfo{
      {append(origPort.name, suffix), type, ModulePort::Direction::Input},
      newInputs.size(),
      {},
      {},
      origPort.loc};
  newInputs.emplace_back(0, newPort);

  if (!body)
    return {};
  return body->addArgument(type, origPort.loc);
}

void PortConverterImpl::createNewOutput(PortInfo origPort, const Twine &suffix,
                                        Type type, Value output,
                                        PortInfo &newPort) {
  newPort = PortInfo{
      {append(origPort.name, suffix), type, ModulePort::Direction::Output},
      newOutputs.size(),
      {},
      {},
      origPort.loc};
  newOutputs.emplace_back(0, newPort);

  if (!body)
    return;
  newOutputValues.push_back(output);
}

LogicalResult PortConverterImpl::run() {
  ModulePortInfo ports = mod.getPortList();

  bool foundLoweredPorts = false;

  auto createPortLowering = [&](PortInfo port) {
    auto &loweredPorts = port.dir == ModulePort::Direction::Output
                             ? loweredOutputs
                             : loweredInputs;

    auto loweredPort = ssb->build(port);
    if (failed(loweredPort))
      return failure();

    foundLoweredPorts |= !(*loweredPort)->isUntouched();
    loweredPorts.emplace_back(std::move(*loweredPort));

    if (failed(loweredPorts.back()->init()))
      return failure();

    return success();
  };

  // Dispatch the port conversion builder on the I/O of the module.
  for (PortInfo port : ports)
    if (failed(createPortLowering(port)))
      return failure();

  // Bail early if we didn't find anything to convert.
  if (!foundLoweredPorts) {
    // Memory optimization.
    loweredInputs.clear();
    loweredOutputs.clear();
    return success();
  }

  // Lower the ports -- this mutates the body directly and builds the port
  // lists.
  for (auto &lowering : loweredInputs)
    lowering->lowerPort();
  for (auto &lowering : loweredOutputs)
    lowering->lowerPort();

  // Set up vectors to erase _all_ the ports. It's easier to rebuild everything
  // (including the non-ESI ports) than reason about interleaving the newly
  // lowered ESI ports with the non-ESI ports. Also, the 'modifyPorts' method
  // ends up rebuilding the port lists anyway, so this isn't nearly as expensive
  // as it may seem.
  SmallVector<unsigned> inputsToErase(mod.getNumInputs());
  std::iota(inputsToErase.begin(), inputsToErase.end(), 0);
  SmallVector<unsigned> outputsToErase(mod.getNumOutputs());
  std::iota(outputsToErase.begin(), outputsToErase.end(), 0);

  mod.modifyPorts(newInputs, newOutputs, inputsToErase, outputsToErase);

  if (body) {
    // We should only erase the original arguments. New ones were appended with
    // the `createInput` method call.
    body->eraseArguments([&ports](BlockArgument arg) {
      return arg.getArgNumber() < ports.sizeInputs();
    });
    // Set the new operands, overwriting the old ones.
    body->getTerminator()->setOperands(newOutputValues);
  }

  // Rewrite instances pointing to this module.
  for (auto *instance : moduleNode->uses()) {
    hw::HWInstanceLike instanceLike = instance->getInstance();
    if (!instanceLike)
      continue;
    hw::InstanceOp hwInstance = dyn_cast_or_null<hw::InstanceOp>(*instanceLike);
    if (!hwInstance) {
      return instanceLike->emitOpError(
          "This code only converts hw.instance instances - ask your friendly "
          "neighborhood compiler engineers to implement support for something "
          "like an hw::HWMutableInstanceLike interface");
    }
    updateInstance(hwInstance);
  }

  // Memory optimization -- we don't need these anymore.
  newInputs.clear();
  newOutputs.clear();
  newOutputValues.clear();
  return success();
}

void PortConverterImpl::updateInstance(hw::InstanceOp inst) {
  ImplicitLocOpBuilder b(inst.getLoc(), inst);
  BackedgeBuilder beb(b, inst.getLoc());
  ModulePortInfo ports = mod.getPortList();

  // Create backedges for the future instance results so the signal mappers can
  // use the future results as values.
  SmallVector<Backedge> newResults;
  for (PortInfo outputPort : ports.getOutputs())
    newResults.push_back(beb.get(outputPort.type));

  // Map the operands.
  SmallVector<Value> newOperands(ports.sizeInputs(), {});
  for (size_t oldOpIdx = 0, e = inst.getNumOperands(); oldOpIdx < e; ++oldOpIdx)
    loweredInputs[oldOpIdx]->mapInputSignals(
        b, inst, inst->getOperand(oldOpIdx), newOperands, newResults);

  // Map the results.
  for (size_t oldResIdx = 0, e = inst.getNumResults(); oldResIdx < e;
       ++oldResIdx)
    loweredOutputs[oldResIdx]->mapOutputSignals(
        b, inst, inst->getResult(oldResIdx), newOperands, newResults);

  // Clone the instance. We cannot just modifiy the existing one since the
  // result types might have changed types and number of them.
  assert(llvm::none_of(newOperands, [](Value v) { return !v; }));
  b.setInsertionPointAfter(inst);
  auto newInst =
      b.create<InstanceOp>(mod, inst.getInstanceNameAttr(), newOperands,
                           inst.getParameters(), inst.getInnerSymAttr());
  newInst->setDialectAttrs(inst->getDialectAttrs());

  // Assign the backedges to the new results.
  for (auto [idx, be] : llvm::enumerate(newResults))
    be.setValue(newInst.getResult(idx));

  // Erase the old instance.
  inst.erase();
}
