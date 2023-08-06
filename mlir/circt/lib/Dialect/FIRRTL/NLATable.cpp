//===- NLATable.cpp - Non-Local Anchor Table --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/NLATable.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/BuiltinOps.h"

using namespace circt;
using namespace firrtl;

NLATable::NLATable(Operation *operation) {
  if (auto mod = dyn_cast<mlir::ModuleOp>(operation))
    for (auto &op : *mod.getBody())
      if ((operation = dyn_cast<CircuitOp>(&op)))
        break;

  auto circuit = cast<CircuitOp>(operation);
  // We are assuming it's faster to iterate over the top level twice than cache
  // a large number of options.
  for (auto &op : *circuit.getBodyBlock()) {
    if (auto module = dyn_cast<FModuleLike>(op))
      symToOp[module.getModuleNameAttr()] = module;
    if (auto nla = dyn_cast<hw::HierPathOp>(op))
      addNLA(nla);
  }
}

ArrayRef<hw::HierPathOp> NLATable::lookup(StringAttr name) {
  auto iter = nodeMap.find(name);
  if (iter == nodeMap.end())
    return {};
  return iter->second;
}

ArrayRef<hw::HierPathOp> NLATable::lookup(Operation *op) {
  auto name = op->getAttrOfType<StringAttr>("sym_name");
  if (!name)
    return {};
  return lookup(name);
}

hw::HierPathOp NLATable::getNLA(StringAttr name) {
  auto *n = symToOp.lookup(name);
  return dyn_cast_or_null<hw::HierPathOp>(n);
}

FModuleLike NLATable::getModule(StringAttr name) {
  auto *n = symToOp.lookup(name);
  return dyn_cast_or_null<FModuleLike>(n);
}

void NLATable::addNLA(hw::HierPathOp nla) {
  symToOp[nla.getSymNameAttr()] = nla;
  for (auto ent : nla.getNamepath()) {
    if (auto mod = dyn_cast<FlatSymbolRefAttr>(ent))
      nodeMap[mod.getAttr()].push_back(nla);
    else if (auto inr = ent.dyn_cast<hw::InnerRefAttr>())
      nodeMap[inr.getModule()].push_back(nla);
  }
}

void NLATable::erase(hw::HierPathOp nla, SymbolTable *symbolTable) {
  symToOp.erase(nla.getSymNameAttr());
  for (auto ent : nla.getNamepath())
    if (auto mod = dyn_cast<FlatSymbolRefAttr>(ent))
      llvm::erase_value(nodeMap[mod.getAttr()], nla);
    else if (auto inr = ent.dyn_cast<hw::InnerRefAttr>())
      llvm::erase_value(nodeMap[inr.getModule()], nla);
  if (symbolTable)
    symbolTable->erase(nla);
}

void NLATable::updateModuleInNLA(hw::HierPathOp nlaOp, StringAttr oldModule,
                                 StringAttr newModule) {
  nlaOp.updateModule(oldModule, newModule);
  auto &nlas = nodeMap[oldModule];
  auto *iter = std::find(nlas.begin(), nlas.end(), nlaOp);
  if (iter != nlas.end()) {
    nlas.erase(iter);
    if (nlas.empty())
      nodeMap.erase(oldModule);
    nodeMap[newModule].push_back(nlaOp);
  }
}

void NLATable::updateModuleInNLA(StringAttr name, StringAttr oldModule,
                                 StringAttr newModule) {
  auto nlaOp = getNLA(name);
  if (!nlaOp)
    return;
  updateModuleInNLA(nlaOp, oldModule, newModule);
}

void NLATable::renameModule(StringAttr oldModName, StringAttr newModName) {
  auto op = symToOp.find(oldModName);
  if (op == symToOp.end())
    return;
  auto iter = nodeMap.find(oldModName);
  if (iter == nodeMap.end())
    return;
  for (auto nla : iter->second)
    nla.updateModule(oldModName, newModName);
  nodeMap[newModName] = iter->second;
  nodeMap.erase(oldModName);
  symToOp[newModName] = op->second;
  symToOp.erase(oldModName);
}

void NLATable::renameModuleAndInnerRef(
    StringAttr newModName, StringAttr oldModName,
    const DenseMap<StringAttr, StringAttr> &innerSymRenameMap) {

  if (newModName == oldModName)
    return;
  for (auto nla : lookup(oldModName)) {
    nla.updateModuleAndInnerRef(oldModName, newModName, innerSymRenameMap);
    nodeMap[newModName].push_back(nla);
  }
  nodeMap.erase(oldModName);
  return;
}
