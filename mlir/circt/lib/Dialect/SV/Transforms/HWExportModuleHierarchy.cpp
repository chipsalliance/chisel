//===- HWExportModuleHierarchy.cpp - Export Module Hierarchy ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Export the module and instance hierarchy information to JSON. This pass looks
// for modules with the firrtl.moduleHierarchyFile attribute and collects the
// hierarchy starting at those modules. The hierarchy information is then
// encoded as JSON in an sv.verbatim op with the output_file attribute set.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/Namespace.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Support/Path.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace circt;

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

class HWExportModuleHierarchyPass
    : public sv::HWExportModuleHierarchyBase<HWExportModuleHierarchyPass> {

private:
  DenseMap<Operation *, hw::ModuleNamespace> moduleNamespaces;

  void printHierarchy(hw::InstanceOp &inst, SymbolTable &symbolTable,
                      llvm::json::OStream &j,
                      SmallVectorImpl<Attribute> &symbols, unsigned &id);

  void extractHierarchyFromTop(hw::HWModuleOp op, SymbolTable &symbolTable,
                               llvm::raw_ostream &os,
                               SmallVectorImpl<Attribute> &symbols);

  void runOnOperation() override;
};

/// Recursively print the module hierarchy as serialized as JSON.
void HWExportModuleHierarchyPass::printHierarchy(
    hw::InstanceOp &inst, SymbolTable &symbolTable, llvm::json::OStream &j,
    SmallVectorImpl<Attribute> &symbols, unsigned &id) {
  auto moduleOp = inst->getParentOfType<hw::HWModuleOp>();
  auto innerSym = inst.getInnerSymAttr();
  if (!innerSym) {
    if (moduleNamespaces.find(moduleOp) == moduleNamespaces.end())
      moduleNamespaces.insert({moduleOp, hw::ModuleNamespace(moduleOp)});
    hw::ModuleNamespace &ns = moduleNamespaces[moduleOp];
    innerSym = hw::InnerSymAttr::get(
        StringAttr::get(inst.getContext(), ns.newName(inst.getInstanceName())));
    inst->setAttr("inner_sym", innerSym);
  }

  j.object([&] {
    j.attribute("instance_name", ("{{" + Twine(id++) + "}}").str());
    symbols.push_back(hw::InnerRefAttr::get(moduleOp.getModuleNameAttr(),
                                            innerSym.getSymName()));
    j.attribute("module_name", ("{{" + Twine(id++) + "}}").str());
    symbols.push_back(inst.getModuleNameAttr());
    j.attributeArray("instances", [&] {
      // Only recurse on module ops, not extern or generated ops, whose internal
      // are opaque.
      auto *nextModuleOp =
          symbolTable.lookup(inst.getModuleNameAttr().getValue());
      if (auto module = dyn_cast<hw::HWModuleOp>(nextModuleOp)) {
        for (auto op : module.getOps<hw::InstanceOp>()) {
          printHierarchy(op, symbolTable, j, symbols, id);
        }
      }
    });
  });
}

/// Return the JSON-serialized module hierarchy for the given module as the top
/// of the hierarchy.
void HWExportModuleHierarchyPass::extractHierarchyFromTop(
    hw::HWModuleOp op, SymbolTable &symbolTable, llvm::raw_ostream &os,
    SmallVectorImpl<Attribute> &symbols) {
  llvm::json::OStream j(os, 2);

  // As a special case for top-level module, set instance name to module name,
  // since the top-level module is not instantiated.
  j.object([&] {
    j.attribute("instance_name", "{{0}}");
    j.attribute("module_name", "{{0}}");
    symbols.push_back(FlatSymbolRefAttr::get(op.getNameAttr()));
    j.attributeArray("instances", [&] {
      unsigned id = 1;
      for (auto op : op.getOps<hw::InstanceOp>())
        printHierarchy(op, symbolTable, j, symbols, id);
    });
  });
}

/// Find the modules corresponding to the firrtl mainModule and DesignUnderTest,
/// and if they exist, emit a verbatim op with the module hierarchy for each.
void HWExportModuleHierarchyPass::runOnOperation() {
  mlir::ModuleOp mlirModule = getOperation();
  std::optional<SymbolTable *> symbolTable;

  for (auto op : mlirModule.getOps<hw::HWModuleOp>()) {
    auto attr = op->getAttrOfType<ArrayAttr>("firrtl.moduleHierarchyFile");
    if (!attr)
      continue;
    for (auto file : attr.getAsRange<hw::OutputFileAttr>()) {
      if (!symbolTable)
        symbolTable = &getAnalysis<SymbolTable>();

      std::string jsonBuffer;
      llvm::raw_string_ostream os(jsonBuffer);
      SmallVector<Attribute> symbols;

      extractHierarchyFromTop(op, **symbolTable, os, symbols);

      auto builder = ImplicitLocOpBuilder::atBlockEnd(
          UnknownLoc::get(mlirModule.getContext()), mlirModule.getBody());
      auto verbatim = builder.create<sv::VerbatimOp>(
          jsonBuffer, ValueRange{}, builder.getArrayAttr(symbols));
      verbatim->setAttr("output_file", file);
    }
  }

  markAllAnalysesPreserved();
}

//===----------------------------------------------------------------------===//
// Pass Creation
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::Pass>
sv::createHWExportModuleHierarchyPass(std::optional<std::string> directory) {
  return std::make_unique<HWExportModuleHierarchyPass>();
}
