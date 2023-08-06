//===- RemoveUnusedPorts.cpp - Remove Dead Ports ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "firrtl-remove-unused-ports"

using namespace circt;
using namespace firrtl;

namespace {
struct RemoveUnusedPortsPass
    : public RemoveUnusedPortsBase<RemoveUnusedPortsPass> {
  void runOnOperation() override;
  void removeUnusedModulePorts(FModuleOp module,
                               InstanceGraphNode *instanceGraphNode);

  /// If true, the pass will remove unused ports even if they have carry a
  /// symbol or annotations. This is likely to break the IR, but may be useful
  /// for `circt-reduce` where preserving functional correctness of the IR is
  /// not important.
  bool ignoreDontTouch = false;
};
} // namespace

void RemoveUnusedPortsPass::runOnOperation() {
  auto &instanceGraph = getAnalysis<InstanceGraph>();
  LLVM_DEBUG(llvm::dbgs() << "===----- Remove unused ports -----==="
                          << "\n");
  // Iterate in the reverse order of instance graph iterator, i.e. from leaves
  // to top.
  for (auto *node : llvm::post_order(&instanceGraph))
    if (auto module = dyn_cast<FModuleOp>(*node->getModule()))
      // Don't prune the main module.
      if (!module.isPublic())
        removeUnusedModulePorts(module, node);
}

void RemoveUnusedPortsPass::removeUnusedModulePorts(
    FModuleOp module, InstanceGraphNode *instanceGraphNode) {
  LLVM_DEBUG(llvm::dbgs() << "Prune ports of module: " << module.getName()
                          << "\n");
  // This tracks constant values of output ports. None indicates an invalid
  // value.
  SmallVector<std::optional<APSInt>> outputPortConstants;
  auto ports = module.getPorts();
  // This tracks port indexes that can be erased.
  llvm::BitVector removalPortIndexes(ports.size());

  for (const auto &e : llvm::enumerate(ports)) {
    unsigned index = e.index();
    auto port = e.value();
    auto arg = module.getArgument(index);

    // If the port is don't touch or has unprocessed annotations, we cannot
    // remove the port. Maybe we can allow annotations though.
    if ((hasDontTouch(arg) || !port.annotations.empty()) && !ignoreDontTouch)
      continue;

    // TODO: Handle inout ports.
    if (port.isInOut())
      continue;

    // If the port is input and has an user, we cannot remove the
    // port.
    if (port.isInput() && !arg.use_empty())
      continue;

    auto portIsUnused = [&](InstanceRecord *a) -> bool {
      auto port = a->getInstance()->getResult(arg.getArgNumber());
      return port.getUses().empty();
    };

    // Output port.
    if (port.isOutput()) {
      if (arg.use_empty()) {
        // Sometimes the connection is already removed possibly by IMCP.
        // In that case, regard the port value as an invalid value.
        outputPortConstants.push_back(std::nullopt);
      } else if (llvm::all_of(instanceGraphNode->uses(), portIsUnused)) {
        // Replace the port with a wire if it is unused.
        auto builder = ImplicitLocOpBuilder::atBlockBegin(
            arg.getLoc(), module.getBodyBlock());
        auto wire = builder.create<WireOp>(arg.getType());
        arg.replaceAllUsesWith(wire.getResult());
        outputPortConstants.push_back(std::nullopt);
      } else if (arg.hasOneUse()) {
        // If the port has a single use, check the port is only connected to
        // invalid or constant
        Operation *op = arg.use_begin().getUser();
        auto connectLike = dyn_cast<FConnectLike>(op);
        if (!connectLike)
          continue;
        auto *srcOp = connectLike.getSrc().getDefiningOp();
        if (!isa_and_nonnull<InvalidValueOp, ConstantOp>(srcOp))
          continue;

        if (auto constant = dyn_cast<ConstantOp>(srcOp))
          outputPortConstants.push_back(constant.getValue());
        else {
          assert(isa<InvalidValueOp>(srcOp) && "only expect invalid");
          outputPortConstants.push_back(std::nullopt);
        }

        // Erase connect op because we are going to remove this output ports.
        op->erase();

        if (srcOp->use_empty())
          srcOp->erase();
      } else {
        // Otherwise, we cannot remove the port.
        continue;
      }
    }

    removalPortIndexes.set(index);
  }

  // If there is nothing to remove, abort.
  if (removalPortIndexes.none())
    return;

  // Delete ports from the module.
  module.erasePorts(removalPortIndexes);
  LLVM_DEBUG(llvm::for_each(removalPortIndexes.set_bits(), [&](unsigned index) {
               llvm::dbgs() << "Delete port: " << ports[index].name << "\n";
             }););

  // Rewrite all uses.
  for (auto *use : instanceGraphNode->uses()) {
    auto instance = ::cast<InstanceOp>(*use->getInstance());
    ImplicitLocOpBuilder builder(instance.getLoc(), instance);
    unsigned outputPortIndex = 0;
    for (auto index : removalPortIndexes.set_bits()) {
      auto result = instance.getResult(index);
      assert(!ports[index].isInOut() && "don't expect inout ports");

      // If the port is input, replace the port with an unwritten wire
      // so that we can remove use-chains in SV dialect canonicalization.
      if (ports[index].isInput()) {
        WireOp wire = builder.create<WireOp>(result.getType());

        // Check that the input port is only written. Sometimes input ports are
        // used as temporary wires. In that case, we cannot erase connections.
        bool onlyWritten = llvm::all_of(result.getUsers(), [&](Operation *op) {
          if (auto connect = dyn_cast<FConnectLike>(op))
            return connect.getDest() == result;
          return false;
        });

        result.replaceUsesWithIf(wire.getResult(), [&](OpOperand &op) -> bool {
          // Connects can be deleted directly.
          if (onlyWritten && isa<FConnectLike>(op.getOwner())) {
            op.getOwner()->erase();
            return false;
          }
          return true;
        });

        // If the wire doesn't have an user, just erase it.
        if (wire.use_empty())
          wire.erase();

        continue;
      }

      // Output port. Replace with the output port with an invalid or constant
      // value.
      auto portConstant = outputPortConstants[outputPortIndex++];
      Value value;
      if (portConstant)
        value = builder.create<ConstantOp>(*portConstant);
      else
        value = builder.create<InvalidValueOp>(result.getType());

      result.replaceAllUsesWith(value);
    }

    // Create a new instance op without unused ports.
    instance.erasePorts(builder, removalPortIndexes);
    // Remove old one.
    instance.erase();
  }

  numRemovedPorts += removalPortIndexes.count();
}

std::unique_ptr<mlir::Pass>
circt::firrtl::createRemoveUnusedPortsPass(bool ignoreDontTouch) {
  auto pass = std::make_unique<RemoveUnusedPortsPass>();
  pass->ignoreDontTouch = ignoreDontTouch;
  return pass;
}
