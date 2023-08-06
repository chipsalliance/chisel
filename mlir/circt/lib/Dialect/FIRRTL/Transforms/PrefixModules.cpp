//===- PrefixModules.cpp - Prefix module names pass -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the PrefixModules pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/NLATable.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/StringMap.h"

using namespace circt;
using namespace firrtl;

/// This maps a FModuleOp to a list of all prefixes that need to be applied.
/// When a module has multiple prefixes, it will be cloned for each one. Usually
/// there is only a single prefix applied to each module, although there could
/// be many.
using PrefixMap = llvm::DenseMap<StringRef, std::vector<std::string>>;

/// Insert a string into the end of vector if the string is not already present.
static void recordPrefix(PrefixMap &prefixMap, StringRef moduleName,
                         std::string prefix) {
  auto &modulePrefixes = prefixMap[moduleName];
  if (llvm::find(modulePrefixes, prefix) == modulePrefixes.end())
    modulePrefixes.push_back(prefix);
}

namespace {
/// This is the prefix which will be applied to a module.
struct PrefixInfo {

  /// The string to prefix on to the module and all of its children.
  StringRef prefix;

  /// If true, this prefix applies to the module itself.  If false, the prefix
  /// only applies to the module's children.
  bool inclusive;
};
} // end anonymous namespace

/// Get the PrefixInfo for a module from a NestedPrefixModulesAnnotation on a
/// module. If the module is not annotated, the prefix returned will be empty.
static PrefixInfo getPrefixInfo(Operation *module) {
  AnnotationSet annotations(module);

  // Get the annotation from the module.
  auto anno = annotations.getAnnotation(prefixModulesAnnoClass);
  if (!anno)
    return {"", false};

  // Get the prefix from the annotation.
  StringRef prefix = "";
  if (auto prefixAttr = anno.getMember<StringAttr>("prefix"))
    prefix = prefixAttr.getValue();

  // Get the inclusive flag from the annotation.
  bool inclusive = false;
  if (auto inclusiveAttr = anno.getMember<BoolAttr>("inclusive"))
    inclusive = inclusiveAttr.getValue();

  return {prefix, inclusive};
}

/// If there is an inclusive prefix attached to the module, return it.
static StringRef getPrefix(Operation *module) {
  auto prefixInfo = getPrefixInfo(module);
  if (prefixInfo.inclusive)
    return prefixInfo.prefix;
  return "";
}

/// This pass finds modules annotated with NestedPrefixAnnotation and prefixes
/// module names using the string stored in the annotation.  This pass prefixes
/// every module instantiated under the annotated root module's hierarchy. If a
/// module is instantiated under two different prefix hierarchies, it will be
/// duplicated and each module will have one prefix applied.
namespace {
class PrefixModulesPass : public PrefixModulesBase<PrefixModulesPass> {
  void removeDeadAnnotations(StringAttr moduleName, Operation *op);
  void renameModuleBody(std::string prefix, StringRef oldName,
                        FModuleOp module);
  void renameModule(FModuleOp module);
  void renameExtModule(FExtModuleOp extModule);
  void renameMemModule(FMemModuleOp memModule);
  void runOnOperation() override;

  /// Mutate Grand Central Interface definitions (an Annotation on the circuit)
  /// with a field "prefix" containing the prefix for that annotation.  This
  /// relies on information built up during renameModule and stored in
  /// interfacePrefixMap.
  void prefixGrandCentralInterfaces();

  /// This is a map from a module name to new prefixes to be applied.
  PrefixMap prefixMap;

  /// A map of Grand Central interface ID to prefix.
  DenseMap<Attribute, std::string> interfacePrefixMap;

  /// Cached instance graph analysis.
  InstanceGraph *instanceGraph = nullptr;

  /// Cached nla table analysis.
  NLATable *nlaTable = nullptr;

  /// Boolean keeping track of any name changes.
  bool anythingChanged = false;
};
} // namespace

/// When a module is cloned, it carries with it all non-local annotations. This
/// function will remove all non-local annotations from the clone with a path
/// that doesn't match.
void PrefixModulesPass::removeDeadAnnotations(StringAttr moduleName,
                                              Operation *op) {
  // A predicate to check if an annotation can be removed. If there is a
  // reference to a NLA, the NLA should either contain this module in its path,
  // if its an InstanceOp. Else, it must exist at the leaf of the NLA. Otherwise
  // the NLA reference can be removed, since its a spurious annotation, result
  // of cloning the original module.
  auto canRemoveAnno = [&](Annotation anno, Operation *op) -> bool {
    auto nla = anno.getMember("circt.nonlocal");
    if (!nla)
      return false;
    auto nlaName = cast<FlatSymbolRefAttr>(nla).getAttr();
    auto nlaOp = nlaTable->getNLA(nlaName);
    if (!nlaOp) {
      op->emitError("cannot find HierPathOp :" + nlaName.getValue());
      signalPassFailure();
      return false;
    }

    bool isValid = false;
    if (isa<InstanceOp>(op))
      isValid = nlaOp.hasModule(moduleName);
    else
      isValid = nlaOp.leafMod() == moduleName;
    return !isValid;
  };
  AnnotationSet::removePortAnnotations(
      op, std::bind(canRemoveAnno, std::placeholders::_2, op));
  AnnotationSet::removeAnnotations(
      op, std::bind(canRemoveAnno, std::placeholders::_1, op));
}

/// Applies the prefix to the module.  This will update the required prefixes of
/// any referenced module in the prefix map.
void PrefixModulesPass::renameModuleBody(std::string prefix, StringRef oldName,
                                         FModuleOp module) {
  auto *context = module.getContext();

  // If we are renaming the body of this module, we need to mark that we have
  // changed the IR. If we are prefixing with the empty string, then nothing has
  // changed yet.
  if (!prefix.empty())
    anythingChanged = true;
  StringAttr thisMod = module.getNameAttr();

  // Remove spurious NLA references from the module ports and the module itself.
  // Some of the NLA references become invalid after a module is cloned, based
  // on the instance.
  removeDeadAnnotations(thisMod, module);

  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement(
      [&](hw::InnerRefAttr innerRef) -> std::pair<Attribute, WalkResult> {
        StringAttr moduleName = innerRef.getModule();
        StringAttr symName = innerRef.getName();

        StringAttr newTarget;
        if (moduleName == oldName) {
          newTarget = module.getNameAttr();
        } else {
          auto target = instanceGraph->lookup(moduleName)->getModule();
          newTarget = StringAttr::get(context, prefix + getPrefix(target) +
                                                   target.getModuleName());
        }
        return {hw::InnerRefAttr::get(newTarget, symName), WalkResult::skip()};
      });

  module.getBody().walk([&](Operation *op) {
    // Remove spurious NLA references either on a leaf op, or the InstanceOp.
    removeDeadAnnotations(thisMod, op);

    if (auto memOp = dyn_cast<MemOp>(op)) {
      StringAttr newPrefix;
      if (auto oldPrefix = memOp->getAttrOfType<StringAttr>("prefix"))
        newPrefix = StringAttr::get(context, prefix + oldPrefix.getValue());
      else
        newPrefix = StringAttr::get(context, prefix);
      memOp->setAttr("prefix", newPrefix);
    } else if (auto instanceOp = dyn_cast<InstanceOp>(op)) {
      auto target = dyn_cast<FModuleLike>(
          *instanceGraph->getReferencedModule(instanceOp));

      // Skip all external modules, unless one of the following conditions
      // is true:
      //   - This is a Grand Central Data Tap
      //   - This is a Grand Central Mem Tap
      if (auto *extModule = dyn_cast_or_null<FExtModuleOp>(&target)) {
        auto isDataTap =
            AnnotationSet(*extModule).hasAnnotation(dataTapsBlackboxClass);
        auto isMemTap = AnnotationSet::forPort(*extModule, 0)
                            .hasAnnotation(memTapPortClass);
        if (!isDataTap && !isMemTap)
          return;
      }

      // Record that we must prefix the target module with the current prefix.
      recordPrefix(prefixMap, target.getModuleName(), prefix);

      // Fixup this instance op to use the prefixed module name.  Note that the
      // referenced FModuleOp will be renamed later.
      auto newTarget = StringAttr::get(context, prefix + getPrefix(target) +
                                                    target.getModuleName());
      AnnotationSet instAnnos(instanceOp);
      // If the instance has HierPathOp, then update its module name also.
      // There can be multiple HierPathOps attached to the instance op.

      StringAttr oldModName = instanceOp.getModuleNameAttr().getAttr();
      // Update the NLAs that apply on this InstanceOp.
      for (Annotation anno : instAnnos) {
        if (auto nla = anno.getMember("circt.nonlocal")) {
          auto nlaName = cast<FlatSymbolRefAttr>(nla).getAttr();
          nlaTable->updateModuleInNLA(nlaName, oldModName, newTarget);
        }
      }
      // Now get the NLAs that pass through the InstanceOp and update them also.
      DenseSet<hw::HierPathOp> instNLAs;
      nlaTable->getInstanceNLAs(instanceOp, instNLAs);
      for (auto nla : instNLAs)
        nlaTable->updateModuleInNLA(nla, oldModName, newTarget);

      instanceOp.setModuleNameAttr(FlatSymbolRefAttr::get(context, newTarget));
    } else {
      replacer.replaceElementsIn(op);
    }
  });
}

/// Apply all required renames to the current module.  This will update the
/// prefix map for any referenced module.
void PrefixModulesPass::renameModule(FModuleOp module) {
  // If the module is annotated to have a prefix, it will be applied after the
  // parent's prefix.
  auto prefixInfo = getPrefixInfo(module);
  auto innerPrefix = prefixInfo.prefix;

  // Remove the annotation from the module.
  AnnotationSet::removeAnnotations(module, prefixModulesAnnoClass);

  // We only add the annotated prefix to the module name if it is inclusive.
  auto oldName = module.getName().str();
  std::string moduleName =
      (prefixInfo.inclusive ? innerPrefix + oldName : oldName).str();

  auto &prefixes = prefixMap[module.getName()];

  // If there are no required prefixes of this module, then this module is a
  // top-level module, and there is an implicit requirement that it has an empty
  // prefix. This empty prefix will be applied to all modules instantiated by
  // this module.
  if (prefixes.empty())
    prefixes.push_back("");

  auto &firstPrefix = prefixes.front();

  auto fixNLAsRootedAt = [&](StringAttr oldModName, StringAttr newModuleName) {
    DenseSet<hw::HierPathOp> nlas;
    nlaTable->getNLAsInModule(oldModName, nlas);
    for (auto n : nlas)
      if (n.root() == oldModName)
        nlaTable->updateModuleInNLA(n, oldModName, newModuleName);
  };
  // Rename the module for each required prefix. This will clone the module
  // once for each prefix but the first.
  OpBuilder builder(module);
  builder.setInsertionPointAfter(module);
  auto oldModName = module.getNameAttr();
  for (auto &outerPrefix : llvm::drop_begin(prefixes)) {
    auto moduleClone = cast<FModuleOp>(builder.clone(*module));
    std::string newModName = outerPrefix + moduleName;
    auto newModNameAttr = StringAttr::get(module.getContext(), newModName);
    moduleClone.setName(newModNameAttr);
    // It is critical to add the new module to the NLATable, otherwise the
    // rename operation would fail.
    nlaTable->addModule(moduleClone);
    fixNLAsRootedAt(oldModName, newModNameAttr);
    // Each call to this function could invalidate the `prefixes` reference.
    renameModuleBody((outerPrefix + innerPrefix).str(), oldName, moduleClone);
  }

  auto prefixFull = (firstPrefix + innerPrefix).str();
  auto newModuleName = firstPrefix + moduleName;
  auto newModuleNameAttr = StringAttr::get(module.getContext(), newModuleName);

  // The first prefix renames the module in place. There is always at least 1
  // prefix.
  module.setName(newModuleNameAttr);
  nlaTable->addModule(module);
  fixNLAsRootedAt(oldModName, newModuleNameAttr);
  renameModuleBody(prefixFull, oldName, module);

  AnnotationSet annotations(module);
  SmallVector<Annotation, 1> newAnnotations;
  annotations.removeAnnotations([&](Annotation anno) {
    if (anno.getClass() == dutAnnoClass) {
      anno.setMember("prefix", builder.getStringAttr(prefixFull));
      newAnnotations.push_back(anno);
      return true;
    }

    // If this module contains a Grand Central interface, then also apply
    // renames to that, but only if there are prefixes to apply.
    if (anno.getClass() == companionAnnoClass)
      interfacePrefixMap[anno.getMember<IntegerAttr>("id")] = prefixFull;
    return false;
  });

  // If any annotations were updated, then update the annotations on the module.
  if (!newAnnotations.empty()) {
    annotations.addAnnotations(newAnnotations);
    annotations.applyToOperation(module);
  }
}

/// Apply prefixes from the `prefixMap` to an external module.  No modifications
/// are made if there are no prefixes for this external module.  If one prefix
/// exists, then the external module will be updated in place.  If multiple
/// prefixes exist, then the original external module will be updated in place
/// and prefixes _after_ the first will cause the module to be cloned
/// ("duplicated" in Scala FIRRTL Compiler terminology).  The logic of this
/// member function is the same as `renameModule` except that there is no module
/// body to recursively update.
void PrefixModulesPass::renameExtModule(FExtModuleOp extModule) {
  // Lookup prefixes for this module.  If none exist, bail out.
  auto &prefixes = prefixMap[extModule.getName()];
  if (prefixes.empty())
    return;

  OpBuilder builder(extModule);
  builder.setInsertionPointAfter(extModule);

  // Function to apply an outer prefix to an external module.  If the module has
  // an optional "defname" (a name that will be used to generate Verilog), also
  // update the defname.
  auto applyPrefixToNameAndDefName = [&](FExtModuleOp &extModule,
                                         StringRef prefix) {
    extModule.setName((prefix + extModule.getName()).str());
    if (auto defname = extModule.getDefname())
      extModule->setAttr("defname", builder.getStringAttr(prefix + *defname));
  };

  // Duplicate the external module if there is more than one prefix.
  for (auto &prefix : llvm::drop_begin(prefixes)) {
    auto duplicate = cast<FExtModuleOp>(builder.clone(*extModule));
    applyPrefixToNameAndDefName(duplicate, prefix);
  }

  // Update the original module with a new prefix.
  applyPrefixToNameAndDefName(extModule, prefixes.front());
}

/// Apply prefixes from the `prefixMap` to a memory module.
void PrefixModulesPass::renameMemModule(FMemModuleOp memModule) {
  // Lookup prefixes for this module.  If none exist, bail out.
  auto &prefixes = prefixMap[memModule.getName()];
  if (prefixes.empty())
    return;

  OpBuilder builder(memModule);
  builder.setInsertionPointAfter(memModule);

  // Duplicate the external module if there is more than one prefix.
  auto originalName = memModule.getName();
  for (auto &prefix : llvm::drop_begin(prefixes)) {
    auto duplicate = cast<FMemModuleOp>(builder.clone(*memModule));
    duplicate.setName((prefix + originalName).str());
    removeDeadAnnotations(duplicate.getNameAttr(), duplicate);
  }

  // Update the original module with a new prefix.
  memModule.setName((prefixes.front() + originalName).str());
  removeDeadAnnotations(memModule.getNameAttr(), memModule);
}

/// Mutate circuit-level annotations to add prefix information to Grand Central
/// (SystemVerilog) interfaces.  Add a "prefix" field to each interface
/// definition (an annotation with class "AugmentedBundleType") that holds the
/// prefix that was determined during runOnModule.  It is assumed that this
/// field did not exist before.
void PrefixModulesPass::prefixGrandCentralInterfaces() {
  // Early exit if no interfaces need prefixes.
  if (interfacePrefixMap.empty())
    return;

  auto circuit = getOperation();
  OpBuilder builder(circuit);

  SmallVector<Attribute> newCircuitAnnotations;
  for (auto anno : AnnotationSet(circuit)) {
    // Only mutate this annotation if it is an AugmentedBundleType and
    // interfacePrefixMap has prefix information for it.
    StringRef prefix;
    if (anno.isClass(augmentedBundleTypeClass)) {
      if (auto id = anno.getMember<IntegerAttr>("id"))
        prefix = interfacePrefixMap[id];
    }

    // Nothing to do.  Copy the annotation.
    if (prefix.empty()) {
      newCircuitAnnotations.push_back(anno.getDict());
      continue;
    }

    // Add a "prefix" field with the prefix for this interface.  This is safe to
    // put at the back and do a `getWithSorted` because the last field is
    // conveniently called "name".
    NamedAttrList newAnno(anno.getDict().getValue());
    newAnno.append("prefix", builder.getStringAttr(prefix));
    newCircuitAnnotations.push_back(
        DictionaryAttr::getWithSorted(builder.getContext(), newAnno));
  }

  // Overwrite the old circuit annotation with the new one created here.
  AnnotationSet(newCircuitAnnotations, builder.getContext())
      .applyToOperation(circuit);
}

void PrefixModulesPass::runOnOperation() {
  auto *context = &getContext();
  instanceGraph = &getAnalysis<InstanceGraph>();
  nlaTable = &getAnalysis<NLATable>();
  auto circuitOp = getOperation();

  // If the main module is prefixed, we have to update the CircuitOp.
  auto mainModule = instanceGraph->getTopLevelModule();
  auto prefix = getPrefix(mainModule);
  if (!prefix.empty()) {
    auto oldModName = mainModule.getModuleNameAttr();
    auto newMainModuleName =
        StringAttr::get(context, (prefix + circuitOp.getName()).str());
    circuitOp.setNameAttr(newMainModuleName);

    // Now update all the NLAs that have the top level module symbol.
    nlaTable->renameModule(oldModName, newMainModuleName);
    for (auto n : nlaTable->lookup(oldModName))
      if (n.root() == oldModName)
        nlaTable->updateModuleInNLA(n, oldModName, newMainModuleName);
  }

  // Walk all Modules in a top-down order.  For each module, look at the list of
  // required prefixes to be applied.
  DenseSet<InstanceGraphNode *> visited;
  for (auto *current : *instanceGraph) {
    for (auto &node : llvm::inverse_post_order_ext(current, visited)) {
      if (auto module = dyn_cast<FModuleOp>(*node->getModule()))
        renameModule(module);
      if (auto extModule = dyn_cast<FExtModuleOp>(*node->getModule()))
        renameExtModule(extModule);
      if (auto memModule = dyn_cast<FMemModuleOp>(*node->getModule()))
        renameMemModule(memModule);
    }
  }

  // Update any interface definitions if needed.
  prefixGrandCentralInterfaces();

  prefixMap.clear();
  interfacePrefixMap.clear();
  if (!anythingChanged)
    markAllAnalysesPreserved();
}

std::unique_ptr<mlir::Pass> circt::firrtl::createPrefixModulesPass() {
  return std::make_unique<PrefixModulesPass>();
}
