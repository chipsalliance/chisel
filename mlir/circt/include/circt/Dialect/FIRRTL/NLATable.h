//===- NLATable.h - Non-Local Anchor Table----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the FIRRTL NLATable.
//
//===----------------------------------------------------------------------===//
#ifndef CIRCT_DIALECT_FIRRTL_NLATABLE_H
#define CIRCT_DIALECT_FIRRTL_NLATABLE_H

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/iterator.h"

namespace circt {
namespace firrtl {

/// This table tracks nlas and what modules participate in them.
///
/// To use this class, retrieve a cached copy from the analysis manager:
///   auto &nlaTable = getAnalysis<NLATable>(getOperation());
class NLATable {

public:
  /// Create a new NLA table of a circuit. This must be called on a FIRRTL
  /// CircuitOp or MLIR ModuleOp. To ensure that the analysis does not return
  /// stale data while a pass is running, it should be kept up-to-date when
  /// modules are added or renamed and NLAs are updated.
  explicit NLATable(Operation *operation);

  /// Lookup all NLAs an operation participates in. This returns a reference to
  /// the internal record, so make a copy before making any update to the
  /// NLATable.
  ArrayRef<hw::HierPathOp> lookup(Operation *op);

  /// Lookup all NLAs an operation participates in. This returns a reference to
  /// the internal record, so make a copy before making any update to the
  /// NLATable.
  ArrayRef<hw::HierPathOp> lookup(StringAttr name);

  /// Resolve a symbol to an NLA.
  hw::HierPathOp getNLA(StringAttr name);

  /// Resolve a symbol to a Module.
  FModuleLike getModule(StringAttr name);

  /// Compute the NLAs that are common between the two modules, `mod1` and
  /// `mod2` and insert them into the set `common`.
  ///  The set of NLAs that an instance op participates in is the set of common
  ///  NLAs between the parent module and the instance target. This can be used
  ///  to get the set of NLAs that an InstanceOp participates in, instead of
  ///  recording them on the op in the IR.
  void commonNLAs(StringAttr mod1, StringAttr mod2,
                  DenseSet<hw::HierPathOp> &common) {
    auto mod1NLAs = lookup(mod1);
    auto mod2NLAs = lookup(mod2);
    common.insert(mod1NLAs.begin(), mod1NLAs.end());
    DenseSet<hw::HierPathOp> set2(mod2NLAs.begin(), mod2NLAs.end());
    llvm::set_intersect(common, set2);
  }

  /// Get the NLAs that the InstanceOp participates in, insert it to the
  /// DenseSet `nlas`.
  void getInstanceNLAs(InstanceOp inst, DenseSet<hw::HierPathOp> &nlas) {
    auto instSym = getInnerSymName(inst);
    // If there is no inner sym on the InstanceOp, then it does not participate
    // in any NLA.
    if (!instSym)
      return;
    auto mod = inst->getParentOfType<FModuleOp>().getNameAttr();
    // Get the NLAs that are common between the parent module and the target
    // module. This should contain the NLAs that this InstanceOp participates
    // in.
    commonNLAs(inst->getParentOfType<FModuleOp>().getNameAttr(),
               inst.getModuleNameAttr().getAttr(), nlas);
    // Handle the case when there are more than one Instances for the same
    // target module. Getting the `commonNLA`, in that case is not enough,
    // remove the NLAs that donot have the InstanceOp as the innerSym.
    for (auto nla : llvm::make_early_inc_range(nlas)) {
      if (!nla.hasInnerSym(mod, instSym))
        nlas.erase(nla);
    }
  }

  /// Get the NLAs that the module `modName` particiaptes in, and insert them
  /// into the DenseSet `nlas`.
  void getNLAsInModule(StringAttr modName, DenseSet<hw::HierPathOp> &nlas) {
    for (auto nla : lookup(modName))
      nlas.insert(nla);
  }

  //===-------------------------------------------------------------------------
  // Methods to keep an NLATable up to date.
  //
  // These methods are not thread safe.  Make sure that modifications are
  // properly synchronized or performed in a serial context.  When the
  // NLATable is used as an analysis, this is only safe when the pass is
  // on a CircuitOp.

  /// Insert a new NLA. This updates two internal records,
  /// 1. Update the map for the `nlaOp` name to the Operation.
  /// 2. For each module in the NLA namepath, insert the NLA into the list of
  /// hwHierPathOps that participate in the corresponding module. This does
  /// not update the module name to module op map, if any potentially new module
  /// in the namepath does not already exist in the record.
  void addNLA(hw::HierPathOp nla);

  /// Remove the NLA from the analysis. This updates two internal records,
  /// 1. Remove the NLA name to the operation map entry.
  /// 2. For each module in the namepath of the NLA, remove the entry from the
  ///    list of NLAs that the module participates in.
  /// Note that this invalidates any reference to the NLA list returned by
  /// 'lookup'.
  void erase(hw::HierPathOp nlaOp, SymbolTable *symbolTable = nullptr);

  /// Record a new FModuleLike operation. This updates the Module name to Module
  /// operation map.
  void addModule(FModuleLike mod) { symToOp[mod.getModuleNameAttr()] = mod; }

  /// Stop tracking a module. Remove the module from two internal records,
  /// 1. Module name to Module op map.
  /// 2. Module name to list of NLAs that the module participates in.
  void eraseModule(StringAttr name) {
    symToOp.erase(name);
    nodeMap.erase(name);
  }

  /// Replace the module `oldModule` with `newModule` in the namepath of the nla
  /// `nlaName`. This moves the nla from the list of `oldModule` to `newModule`.
  /// Move `nlaName` from the list of NLAs that `oldModule` participates in to
  /// `newModule`. This can delete and invalidate any reference returned by
  /// `lookup`.
  void updateModuleInNLA(StringAttr nlaName, StringAttr oldModule,
                         StringAttr newModule);

  /// Replace the module `oldModule` with `newModule` in the namepath of the nla
  /// `nlaOp`. This moves the nla from the list of `oldModule` to `newModule`.
  /// Move `nlaOp` from the list of NLAs that `oldModule` participates in to
  /// `newModule`. This can delete and invalidate any reference returned by
  /// `lookup`.
  void updateModuleInNLA(hw::HierPathOp nlaOp, StringAttr oldModule,
                         StringAttr newModule);

  /// Rename a module, this updates the name to module tracking and the name to
  /// NLA tracking. This moves all the NLAs that `oldModName` is participating
  /// in to the `newModName`. The `oldModName` must exist in the name to module
  /// record. This also removes all the entries for `oldModName`.
  void renameModule(StringAttr oldModName, StringAttr newModName);

  /// Replace the module `oldModName` with `newModName` in the namepath of any
  /// NLA. Since the module is being updated, the symbols inside the module
  /// should also be renamed. Use the rename map `innerSymRenameMap` to update
  /// the inner_sym names in the namepath.
  void renameModuleAndInnerRef(
      StringAttr newModName, StringAttr oldModName,
      const DenseMap<StringAttr, StringAttr> &innerSymRenameMap);

  /// Remove the NLA from the Module. This updates the module name to NLA
  /// tracking.
  void removeNLAfromModule(hw::HierPathOp nla, StringAttr mod) {
    llvm::erase_value(nodeMap[mod], nla);
  }

  /// Remove all the nlas in the set `nlas` from the module. This updates the
  /// module name to NLA tracking.
  void removeNLAsfromModule(const DenseSet<hw::HierPathOp> &nlas,
                            StringAttr mod) {
    llvm::erase_if(nodeMap[mod],
                   [&nlas](const auto &nla) { return nlas.count(nla); });
  }

  /// Add the nla to the module. This ensures that the list of NLAs that the
  /// module participates in is updated. This will be required if `mod` is added
  /// to the namepath of `nla`.
  void addNLAtoModule(hw::HierPathOp nla, StringAttr mod) {
    nodeMap[mod].push_back(nla);
  }

private:
  NLATable(const NLATable &) = delete;

  /// Map modules to the NLA's that target them.
  llvm::DenseMap<StringAttr, SmallVector<hw::HierPathOp, 4>> nodeMap;

  /// Map symbol names to module and NLA operations.
  llvm::DenseMap<StringAttr, Operation *> symToOp;
};

} // namespace firrtl
} // namespace circt
#endif // CIRCT_DIALECT_FIRRTL_NLATABLE_H
