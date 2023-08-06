//===- LegalizeNames.cpp - Name Legalization for ExportVerilog ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This renames modules and variables to avoid conflicts with keywords and other
// declarations.
//
//===----------------------------------------------------------------------===//

#include "ExportVerilogInternals.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Support/LoweringOptions.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace sv;
using namespace hw;
using namespace ExportVerilog;

//===----------------------------------------------------------------------===//
// NameCollisionResolver
//===----------------------------------------------------------------------===//

/// Given a name that may have collisions or invalid symbols, return a
/// replacement name to use, or null if the original name was ok.
StringRef NameCollisionResolver::getLegalName(StringRef originalName) {
  return legalizeName(originalName, nextGeneratedNameIDs);
}

//===----------------------------------------------------------------------===//
// FieldNameResolver
//===----------------------------------------------------------------------===//

void FieldNameResolver::setRenamedFieldName(StringAttr fieldName,
                                            StringAttr newFieldName) {
  renamedFieldNames[fieldName] = newFieldName;
  nextGeneratedNameIDs.insert({newFieldName, 0});
}

StringAttr FieldNameResolver::getRenamedFieldName(StringAttr fieldName) {
  auto it = renamedFieldNames.find(fieldName);
  if (it != renamedFieldNames.end())
    return it->second;

  // If a field name is not verilog name or used already, we have to rename it.
  bool hasToBeRenamed = !sv::isNameValid(fieldName.getValue()) ||
                        nextGeneratedNameIDs.count(fieldName.getValue());

  if (!hasToBeRenamed) {
    setRenamedFieldName(fieldName, fieldName);
    return fieldName;
  }

  StringRef newFieldName =
      sv::legalizeName(fieldName.getValue(), nextGeneratedNameIDs);

  auto newFieldNameAttr = StringAttr::get(fieldName.getContext(), newFieldName);

  setRenamedFieldName(fieldName, newFieldNameAttr);
  return newFieldNameAttr;
}

std::string FieldNameResolver::getEnumFieldName(hw::EnumFieldAttr attr) {
  auto aliasType = attr.getType().getValue().dyn_cast<hw::TypeAliasType>();
  if (!aliasType)
    return attr.getField().getValue().str();

  auto fieldStr = attr.getField().getValue().str();
  if (auto prefix = globalNames.getEnumPrefix(aliasType))
    return (prefix.getValue() + "_" + fieldStr).str();

  // No prefix registered, just use the bare field name.
  return fieldStr;
}

//===----------------------------------------------------------------------===//
// GlobalNameResolver
//===----------------------------------------------------------------------===//

namespace circt {
namespace ExportVerilog {
/// This class keeps track of modules and interfaces that need to be renamed, as
/// well as module ports, parameters, declarations and verif labels that need to
/// be renamed. This can happen either due to conflicts between them or due to
/// a conflict with a Verilog keyword.
///
/// Once constructed, this is immutable.
class GlobalNameResolver {
public:
  /// Construct a GlobalNameResolver and perform name legalization of the
  /// module/interfaces, port/parameter and declaration names.
  GlobalNameResolver(mlir::ModuleOp topLevel, const LoweringOptions &options);

  GlobalNameTable takeGlobalNameTable() { return std::move(globalNameTable); }

private:
  /// Check to see if the port names of the specified module conflict with
  /// keywords or themselves.  If so, add the replacement names to
  /// globalNameTable.
  void legalizeModuleNames(HWModuleOp module);
  void legalizeInterfaceNames(InterfaceOp interface);

  // Gathers prefixes of enum types by inspecting typescopes in the module.
  void gatherEnumPrefixes(mlir::ModuleOp topLevel);

  /// Set of globally visible names, to ensure uniqueness.
  NameCollisionResolver globalNameResolver;

  /// This keeps track of globally visible names like module parameters.
  GlobalNameTable globalNameTable;

  GlobalNameResolver(const GlobalNameResolver &) = delete;
  void operator=(const GlobalNameResolver &) = delete;
};
} // namespace ExportVerilog
} // namespace circt

// This function legalizes local names in the given module.
static void legalizeModuleLocalNames(HWModuleOp module,
                                     const LoweringOptions &options,
                                     const GlobalNameTable &globalNameTable) {
  // A resolver for a local name collison.
  NameCollisionResolver nameResolver;
  // Register names used by parameters.
  for (auto param : module.getParameters())
    nameResolver.insertUsedName(globalNameTable.getParameterVerilogName(
        module, param.cast<ParamDeclAttr>().getName()));

  auto *ctxt = module.getContext();

  auto verilogNameAttr = StringAttr::get(ctxt, "hw.verilogName");
  // Legalize the port names.
  auto ports = module.getPortList();
  for (const PortInfo &port : ports) {
    auto newName = nameResolver.getLegalName(port.name);
    if (newName != port.name.getValue()) {
      if (port.isOutput())
        module.setResultAttr(port.argNum, verilogNameAttr,
                             StringAttr::get(ctxt, newName));
      else
        module.setArgAttr(port.argNum, verilogNameAttr,
                          StringAttr::get(ctxt, newName));
    }
  }

  SmallVector<std::pair<Operation *, StringAttr>> nameEntries;
  // Legalize the value names. We first mark existing hw.verilogName attrs as
  // being used, and then resolve names of declarations.
  module.walk([&](Operation *op) {
    if (!isa<HWModuleOp>(op)) {
      // If there is a hw.verilogName attr, mark names as used.
      if (auto name = op->getAttrOfType<StringAttr>(verilogNameAttr)) {
        nameResolver.insertUsedName(
            op->getAttrOfType<StringAttr>(verilogNameAttr));
      } else if (isa<sv::WireOp, hw::WireOp, RegOp, LogicOp, LocalParamOp,
                     hw::InstanceOp, sv::InterfaceInstanceOp, sv::GenerateOp>(
                     op)) {
        // Otherwise, get a verilog name via `getSymOpName`.
        nameEntries.emplace_back(
            op, StringAttr::get(op->getContext(), getSymOpName(op)));
      } else if (auto forOp = dyn_cast<ForOp>(op)) {
        nameEntries.emplace_back(op, forOp.getInductionVarNameAttr());
      } else if (isa<AssertOp, AssumeOp, CoverOp, AssertConcurrentOp,
                     AssumeConcurrentOp, CoverConcurrentOp, verif::AssertOp,
                     verif::CoverOp, verif::AssumeOp>(op)) {
        // Notice and renamify the labels on verification statements.
        if (auto labelAttr = op->getAttrOfType<StringAttr>("label"))
          nameEntries.emplace_back(op, labelAttr);
        else if (options.enforceVerifLabels) {
          // If labels are required for all verif statements, get a default
          // name from verificaiton kinds.
          StringRef defaultName =
              llvm::TypeSwitch<Operation *, StringRef>(op)
                  .Case<AssertOp, AssertConcurrentOp, verif::AssertOp>(
                      [](auto) { return "assert"; })
                  .Case<CoverOp, CoverConcurrentOp, verif::CoverOp>(
                      [](auto) { return "cover"; })
                  .Case<AssumeOp, AssumeConcurrentOp, verif::AssumeOp>(
                      [](auto) { return "assume"; });
          nameEntries.emplace_back(
              op, StringAttr::get(op->getContext(), defaultName));
        }
      }
    }
  });

  for (auto [op, nameAttr] : nameEntries) {
    auto newName = nameResolver.getLegalName(nameAttr);
    assert(!newName.empty() && "must have a valid name");
    // Add a legalized name to "hw.verilogName" attribute.
    op->setAttr(verilogNameAttr, nameAttr.getValue() == newName
                                     ? nameAttr
                                     : StringAttr::get(ctxt, newName));
  }
}

/// Construct a GlobalNameResolver and do the initial scan to populate and
/// unique the module/interfaces and port/parameter names.
GlobalNameResolver::GlobalNameResolver(mlir::ModuleOp topLevel,
                                       const LoweringOptions &options) {
  // Register the names of external modules which we cannot rename. This has to
  // occur in a first pass separate from the modules and interfaces which we are
  // actually allowed to rename, in order to ensure that we don't accidentally
  // rename a module that later collides with an extern module.
  for (auto &op : *topLevel.getBody()) {
    // Note that external modules *often* have name collisions, because they
    // correspond to the same verilog module with different parameters.
    if (isa<HWModuleExternOp>(op) || isa<HWModuleGeneratedOp>(op)) {
      auto name = getVerilogModuleNameAttr(&op).getValue();
      if (!sv::isNameValid(name))
        op.emitError("name \"")
            << name << "\" is not allowed in Verilog output";
      globalNameResolver.insertUsedName(name);
    }
  }

  // Legalize module and interface names.
  for (auto &op : *topLevel.getBody()) {
    if (auto module = dyn_cast<HWModuleOp>(op)) {
      legalizeModuleNames(module);
      continue;
    }

    // Legalize the name of the interface itself, as well as any signals and
    // modports within it.
    if (auto interface = dyn_cast<InterfaceOp>(op)) {
      legalizeInterfaceNames(interface);
      continue;
    }
  }

  // Legalize names in HW modules parallelly.
  mlir::parallelForEach(
      topLevel.getContext(), topLevel.getOps<HWModuleOp>(), [&](auto module) {
        legalizeModuleLocalNames(module, options, globalNameTable);
      });

  // Gather enum prefixes.
  gatherEnumPrefixes(topLevel);
}

// Gathers prefixes of enum types by investigating typescopes in the module.
void GlobalNameResolver::gatherEnumPrefixes(mlir::ModuleOp topLevel) {
  auto *ctx = topLevel.getContext();
  for (auto typeScope : topLevel.getOps<hw::TypeScopeOp>()) {
    for (auto typeDecl : typeScope.getOps<hw::TypedeclOp>()) {
      auto enumType = typeDecl.getType().dyn_cast<hw::EnumType>();
      if (!enumType)
        continue;

      // Register the enum type as the alias type of the typedecl, since this is
      // how users will request the prefix.
      globalNameTable.enumPrefixes[typeDecl.getAliasType()] =
          StringAttr::get(ctx, typeDecl.getPreferredName());
    }
  }
}

/// Check to see if the port names of the specified module conflict with
/// keywords or themselves.  If so, add the replacement names to
/// globalNameTable.
void GlobalNameResolver::legalizeModuleNames(HWModuleOp module) {
  MLIRContext *ctxt = module.getContext();
  // If the module's symbol itself conflicts, then set a "verilogName" attribute
  // on the module to reflect the name we need to use.
  StringRef oldName = module.getName();
  auto newName = globalNameResolver.getLegalName(oldName);
  if (newName != oldName)
    module->setAttr("verilogName", StringAttr::get(ctxt, newName));

  NameCollisionResolver nameResolver;
  // Legalize the parameter names.
  for (auto param : module.getParameters()) {
    auto paramAttr = param.cast<ParamDeclAttr>();
    auto newName = nameResolver.getLegalName(paramAttr.getName());
    if (newName != paramAttr.getName().getValue())
      globalNameTable.addRenamedParam(module, paramAttr.getName(), newName);
  }
}

void GlobalNameResolver::legalizeInterfaceNames(InterfaceOp interface) {
  MLIRContext *ctxt = interface.getContext();
  auto verilogNameAttr = StringAttr::get(ctxt, "hw.verilogName");
  auto newName = globalNameResolver.getLegalName(interface.getName());
  if (newName != interface.getName())
    interface->setAttr(verilogNameAttr, StringAttr::get(ctxt, newName));

  NameCollisionResolver localNames;
  // Rename signals and modports.
  for (auto &op : *interface.getBodyBlock()) {
    if (isa<InterfaceSignalOp, InterfaceModportOp>(op)) {
      auto name = SymbolTable::getSymbolName(&op).getValue();
      auto newName = localNames.getLegalName(name);
      if (newName != name)
        op.setAttr(verilogNameAttr, StringAttr::get(ctxt, newName));
    }
  }
}

//===----------------------------------------------------------------------===//
// Public interface
//===----------------------------------------------------------------------===//

/// Rewrite module names and interfaces to not conflict with each other or with
/// Verilog keywords.
GlobalNameTable
ExportVerilog::legalizeGlobalNames(ModuleOp topLevel,
                                   const LoweringOptions &options) {
  GlobalNameResolver resolver(topLevel, options);
  return resolver.takeGlobalNameTable();
}
