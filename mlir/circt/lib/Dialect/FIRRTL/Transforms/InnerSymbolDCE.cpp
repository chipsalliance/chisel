//===- InnerSymbolDCE.cpp - Delete Unused Inner Symbols----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This pass removes inner symbols which have no uses.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Threading.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "firrtl-inner-symbol-dce"

using namespace mlir;
using namespace circt;
using namespace firrtl;
using namespace hw;

/// Drop the specified symbol.
/// Belongs in InnerSymbolTable, need to move port symbols accessors
/// into HW(ModuleLike) or perhaps a new inner-symbol interface that
/// dialects can optionally elect to use on their "ModuleLike"s.
/// For now, since InnerSymbolDCE is FIRRTL-only, define this here.
static void dropSymbol(const InnerSymTarget &target) {
  assert(target);
  assert(InnerSymbolTable::getInnerSymbol(target));

  if (target.isPort()) {
    auto mod = cast<HWModuleLike>(target.getOp());
    assert(target.getPort() < mod.getNumPorts());
    auto base = mod.getPortSymbolAttr(target.getPort());
    cast<firrtl::FModuleLike>(*mod).setPortSymbolsAttr(
        target.getPort(), base.erase(target.getField()));
    return;
  }

  auto symOp = cast<InnerSymbolOpInterface>(target.getOp());
  auto base = symOp.getInnerSymAttr();
  symOp.setInnerSymbolAttr(base.erase(target.getField()));
}

struct InnerSymbolDCEPass : public InnerSymbolDCEBase<InnerSymbolDCEPass> {
  void runOnOperation() override;

private:
  void findInnerRefs(Attribute attr);
  void insertInnerRef(InnerRefAttr innerRef);
  void removeInnerSyms(FModuleLike mod);

  DenseSet<std::pair<StringAttr, StringAttr>> innerRefs;
};

/// Find all InnerRefAttrs inside a given Attribute.
void InnerSymbolDCEPass::findInnerRefs(Attribute attr) {
  // Check if this Attribute or any sub-Attributes are InnerRefAttrs.
  attr.walk([&](Attribute subAttr) {
    if (auto innerRef = dyn_cast<InnerRefAttr>(subAttr))
      insertInnerRef(innerRef);
  });
}

/// Add an InnerRefAttr to the set of all InnerRefAttrs.
void InnerSymbolDCEPass::insertInnerRef(InnerRefAttr innerRef) {
  StringAttr moduleName = innerRef.getModule();
  StringAttr symName = innerRef.getName();

  // Track total inner refs found.
  ++numInnerRefsFound;

  auto [iter, inserted] = innerRefs.insert({moduleName, symName});
  if (!inserted)
    return;

  LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << ": found reference to " << moduleName
                          << "::" << symName << '\n';);
}

/// Remove all dead inner symbols from the specified module.
void InnerSymbolDCEPass::removeInnerSyms(FModuleLike mod) {
  auto moduleName = mod.getModuleNameAttr();

  // Walk inner symbols, removing any not referenced.
  InnerSymbolTable::walkSymbols(
      mod, [&](StringAttr name, const InnerSymTarget &target) {
        ++numInnerSymbolsFound;

        // Check if the name is referenced by any InnerRef.
        if (innerRefs.contains({moduleName, name}))
          return;

        dropSymbol(target);
        ++numInnerSymbolsRemoved;

        LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << ": removed " << moduleName
                                << "::" << name << '\n';);
      });
}

void InnerSymbolDCEPass::runOnOperation() {
  // Run on the top-level ModuleOp to include any VerbatimOps that aren't
  // wrapped in a CircuitOp.
  ModuleOp topModule = getOperation();

  // Traverse the entire IR once.
  SmallVector<FModuleLike> modules;
  topModule.walk([&](Operation *op) {
    // Find all InnerRefAttrs.
    for (NamedAttribute namedAttr : op->getAttrs())
      findInnerRefs(namedAttr.getValue());

    // Collect all FModuleLike operations.
    if (auto mod = dyn_cast<FModuleLike>(op))
      modules.push_back(mod);
  });

  // Traverse all FModuleOps in parallel, removing any InnerSymAttrs that are
  // dead code.
  parallelForEach(&getContext(), modules,
                  [&](FModuleLike mod) { removeInnerSyms(mod); });
}

std::unique_ptr<mlir::Pass> circt::firrtl::createInnerSymbolDCEPass() {
  return std::make_unique<InnerSymbolDCEPass>();
}
