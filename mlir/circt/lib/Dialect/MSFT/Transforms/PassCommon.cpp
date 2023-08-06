//===- PassCommon.cpp - PassCommon ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/MSFT/MSFTPasses.h"

using namespace mlir;
using namespace circt;
using namespace msft;

bool circt::msft::isAnyModule(Operation *module) {
  return isa<MSFTModuleOp, MSFTModuleExternOp>(module) ||
         hw::isAnyModule(module);
}

SmallVector<unsigned> circt::msft::makeSequentialRange(unsigned size) {
  SmallVector<unsigned> seq;
  for (size_t i = 0; i < size; ++i)
    seq.push_back(i);
  return seq;
}

StringRef circt::msft::getValueName(Value v, const SymbolCache &syms,
                                    std::string &buff) {
  Operation *defOp = v.getDefiningOp();
  if (auto inst = dyn_cast_or_null<InstanceOp>(defOp)) {
    Operation *modOp = syms.getDefinition(inst.getModuleNameAttr());
    if (modOp) { // If modOp isn't in the cache, it's probably a new module;
      assert(isAnyModule(modOp) && "Instance must point to a module");
      OpResult instResult = v.cast<OpResult>();
      auto mod = cast<hw::HWModuleLike>(modOp);
      buff.clear();
      llvm::raw_string_ostream os(buff);
      os << inst.getInstanceName() << ".";
      StringAttr name = mod.getOutputNameAttr(instResult.getResultNumber());
      if (name)
        os << name.getValue();
      return buff;
    }
  }
  if (auto blockArg = v.dyn_cast<BlockArgument>()) {
    auto portInfo =
        cast<hw::PortList>(blockArg.getOwner()->getParent()->getParentOp())
            .getPortList();
    return portInfo.atInput(blockArg.getArgNumber()).getName();
  }
  if (auto constOp = dyn_cast<hw::ConstantOp>(defOp)) {
    buff.clear();
    llvm::raw_string_ostream(buff) << "c" << constOp.getValue();
    return buff;
  }

  return "";
}

void PassCommon::getAndSortModules(ModuleOp topMod,
                                   SmallVectorImpl<hw::HWModuleLike> &mods) {
  // Add here _before_ we go deeper to prevent infinite recursion.
  DenseSet<Operation *> modsSeen;
  mods.clear();
  moduleInstantiations.clear();
  topMod.walk([&](hw::HWModuleLike mod) {
    getAndSortModulesVisitor(mod, mods, modsSeen);
  });
}

LogicalResult PassCommon::verifyInstances(mlir::ModuleOp mod) {
  WalkResult r = mod.walk([&](InstanceOp inst) {
    Operation *modOp = topLevelSyms.getDefinition(inst.getModuleNameAttr());
    if (!isAnyModule(modOp))
      return WalkResult::interrupt();

    hw::ModulePortInfo ports = cast<hw::PortList>(modOp).getPortList();
    return succeeded(inst.verifySignatureMatch(ports))
               ? WalkResult::advance()
               : WalkResult::interrupt();
  });
  return failure(r.wasInterrupted());
}

// Run a post-order DFS.
void PassCommon::getAndSortModulesVisitor(
    hw::HWModuleLike mod, SmallVectorImpl<hw::HWModuleLike> &mods,
    DenseSet<Operation *> &modsSeen) {
  if (modsSeen.contains(mod))
    return;
  modsSeen.insert(mod);

  mod.walk([&](hw::HWInstanceLike inst) {
    Operation *modOp =
        topLevelSyms.getDefinition(inst.getReferencedModuleNameAttr());
    assert(modOp);
    moduleInstantiations[modOp].push_back(inst);
    if (auto modLike = dyn_cast<hw::HWModuleLike>(modOp))
      getAndSortModulesVisitor(modLike, mods, modsSeen);
  });

  mods.push_back(mod);
}
