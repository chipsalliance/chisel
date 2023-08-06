//===- AddTaps.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/SV/SVOps.h"

using namespace circt;
using namespace arc;
using namespace hw;

namespace {
struct AddTapsPass : public AddTapsBase<AddTapsPass> {
  void runOnOperation() override {
    getOperation().walk([&](Operation *op) {
      TypeSwitch<Operation *>(op)
          .Case<HWModuleOp, sv::WireOp, hw::WireOp>([&](auto op) { tap(op); })
          .Default([&](auto) { tapIfNamed(op); });
    });
  }

  // Add taps for all module ports.
  void tap(HWModuleOp moduleOp) {
    if (!tapPorts)
      return;
    auto *outputOp = moduleOp.getBodyBlock()->getTerminator();
    ModulePortInfo ports = moduleOp.getPortList();

    // Add taps to inputs.
    auto builder = OpBuilder::atBlockBegin(moduleOp.getBodyBlock());
    for (auto [port, arg] :
         llvm::zip(ports.getInputs(), moduleOp.getArguments()))
      builder.create<arc::TapOp>(arg.getLoc(), arg, port.getName());

    // Add taps to outputs.
    builder.setInsertionPoint(outputOp);
    for (auto [port, result] :
         llvm::zip(ports.getOutputs(), outputOp->getOperands()))
      builder.create<arc::TapOp>(result.getLoc(), result, port.getName());
  }

  // Add taps for SV wires.
  void tap(sv::WireOp wireOp) {
    if (!tapWires)
      return;
    sv::ReadInOutOp readOp;
    for (auto *user : wireOp->getUsers())
      if (auto op = dyn_cast<sv::ReadInOutOp>(user))
        readOp = op;

    OpBuilder builder(wireOp);
    if (!readOp)
      readOp = builder.create<sv::ReadInOutOp>(wireOp.getLoc(), wireOp);
    builder.create<arc::TapOp>(readOp.getLoc(), readOp, wireOp.getName());
  }

  // Add taps for HW wires.
  void tap(hw::WireOp wireOp) {
    if (!tapWires)
      return;
    if (auto name = wireOp.getName()) {
      OpBuilder builder(wireOp);
      builder.create<arc::TapOp>(wireOp.getLoc(), wireOp, *name);
    }
  }

  // Add taps for named values.
  void tapIfNamed(Operation *op) {
    if (!tapNamedValues || op->getNumResults() != 1)
      return;
    if (auto name = op->getAttrOfType<StringAttr>("sv.namehint")) {
      OpBuilder builder(op);
      builder.create<arc::TapOp>(op->getLoc(), op->getResult(0), name);
    }
  }

  using AddTapsBase::tapNamedValues;
  using AddTapsBase::tapPorts;
  using AddTapsBase::tapWires;
};
} // namespace

std::unique_ptr<Pass>
arc::createAddTapsPass(std::optional<bool> tapPorts,
                       std::optional<bool> tapWires,
                       std::optional<bool> tapNamedValues) {
  auto pass = std::make_unique<AddTapsPass>();
  if (tapPorts)
    pass->tapPorts = *tapPorts;
  if (tapWires)
    pass->tapWires = *tapWires;
  if (tapNamedValues)
    pass->tapNamedValues = *tapNamedValues;
  return pass;
}
