//===- MSFTDiscoverAppIDs.cpp - App ID discovery pass -----------*- C++ -*-===//
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
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace circt;
using namespace msft;

//===----------------------------------------------------------------------===//
// Discover AppIDs pass
//===----------------------------------------------------------------------===//

namespace {
struct DiscoverAppIDsPass : public DiscoverAppIDsBase<DiscoverAppIDsPass>,
                            MSFTPassCommon {
  void runOnOperation() override;
  void processMod(MSFTModuleOp);
};
} // anonymous namespace

void DiscoverAppIDsPass::runOnOperation() {
  ModuleOp topMod = getOperation();
  topLevelSyms.addDefinitions(topMod);
  if (failed(verifyInstances(topMod))) {
    signalPassFailure();
    return;
  }

  // Sort modules in partial order be use. Enables single-pass processing.
  SmallVector<MSFTModuleOp> sortedMods;
  getAndSortModules(topMod, sortedMods);

  for (MSFTModuleOp mod : sortedMods)
    processMod(mod);
}

/// Find the AppIDs in a given module.
void DiscoverAppIDsPass::processMod(MSFTModuleOp mod) {
  SmallDenseMap<StringAttr, uint64_t> appBaseCounts;
  SmallPtrSet<StringAttr, 32> localAppIDBases;
  SmallDenseMap<AppIDAttr, Operation *> localAppIDs;

  mod.walk([&](Operation *op) {
    // If an operation has an "appid" dialect attribute, it is considered a
    // "local" appid.
    if (auto appid = op->getAttrOfType<AppIDAttr>("msft.appid")) {
      if (localAppIDs.find(appid) != localAppIDs.end()) {
        op->emitOpError("Found multiple identical AppIDs in same module")
                .attachNote(localAppIDs[appid]->getLoc())
            << "first AppID located here";
        signalPassFailure();
      } else {
        localAppIDs[appid] = op;
      }
      localAppIDBases.insert(appid.getName());
    }

    // Instance ops should expose their module's AppIDs recursively. Track the
    // number of instances which contain a base name.
    if (auto inst = dyn_cast<InstanceOp>(op)) {
      auto targetMod = dyn_cast<MSFTModuleOp>(
          topLevelSyms.getDefinition(inst.getModuleNameAttr()));
      if (targetMod && targetMod.getChildAppIDBases())
        for (auto base :
             targetMod.getChildAppIDBasesAttr().getAsRange<StringAttr>())
          appBaseCounts[base] += 1;
    }
  });

  // Collect the list of AppID base names with which to annotate 'mod'.
  SmallVector<Attribute, 32> finalModBases;
  for (auto baseCount : appBaseCounts) {
    // If multiple instances expose the same base name, don't expose them
    // through this module. If any of the instances expose basenames which are
    // exposed locally, also don't expose them up.
    if (baseCount.getSecond() == 1 &&
        !localAppIDBases.contains(baseCount.getFirst()))
      finalModBases.push_back(baseCount.getFirst());
  }

  // Add all of the local base names.
  for (StringAttr lclBase : localAppIDBases)
    finalModBases.push_back(lclBase);

  if (finalModBases.empty())
    return;

  // Sort the list to put it in a reasonable deterministic order.
  llvm::sort(finalModBases, [](Attribute a, Attribute b) {
    return cast<StringAttr>(a).getValue() < cast<StringAttr>(b).getValue();
  });
  ArrayAttr childrenBases = ArrayAttr::get(mod.getContext(), finalModBases);
  mod.setChildAppIDBasesAttr(childrenBases);
}

std::unique_ptr<Pass> circt::msft::createDiscoverAppIDsPass() {
  return std::make_unique<DiscoverAppIDsPass>();
}
