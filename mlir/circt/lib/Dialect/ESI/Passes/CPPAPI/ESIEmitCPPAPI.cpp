//===- ESIEmitCPPAPI.cpp - ESI C++ API emission -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Emit the C++ ESI API.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/ESI/ESIPasses.h"
#include "circt/Dialect/ESI/ESIServices.h"
#include "circt/Dialect/ESI/ESITypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/JSON.h"

#include "CPPAPI.h"

#include <iostream>
#include <memory>

using namespace mlir;
using namespace circt;
using namespace circt::esi;
using namespace cppapi;

namespace {
struct CPPAPI {
  CPPAPI(ModuleOp module, llvm::raw_ostream &os)
      : module(module), os(os), diag(module.getContext()->getDiagEngine()),
        unknown(UnknownLoc::get(module.getContext())) {
    diag.registerHandler([this](Diagnostic &diag) -> LogicalResult {
      if (diag.getSeverity() == mlir::DiagnosticSeverity::Error)
        ++errorCount;
      return failure();
    });
  }

  /// Emit the whole API.
  LogicalResult emit();

private:
  LogicalResult gatherTypes();
  LogicalResult emitTypes();
  LogicalResult emitServiceDeclarations();
  LogicalResult emitDesignModules();
  LogicalResult emitGlobalNamespace();

  ModuleOp module;
  mlir::raw_indented_ostream os;
  mlir::DiagnosticEngine &diag;
  const Location unknown;
  size_t errorCount = 0;

  llvm::MapVector<mlir::Type, CPPType> types;
  llvm::SmallVector<CPPService> cppServices;
};
} // anonymous namespace

LogicalResult CPPAPI::gatherTypes() {
  auto storeType = [&](mlir::Type type) -> LogicalResult {
    auto dirType = esi::innerType(type);
    auto dirTypeSchemaIt = types.find(dirType);
    if (dirTypeSchemaIt == types.end()) {
      CPPType dirTypeSchema(dirType);
      if (!dirTypeSchema.isSupported())
        return emitError(module.getLoc())
               << "Type " << dirType << " not supported.";
      dirTypeSchemaIt = types.insert({dirType, dirTypeSchema}).first;
    }
    return success();
  };

  for (auto serviceDeclOp : module.getOps<ServiceDeclOpInterface>()) {
    llvm::SmallVector<ServicePortInfo> ports;
    serviceDeclOp.getPortList(ports);
    for (auto portInfo : ports) {
      if (portInfo.toClientType)
        if (failed(storeType(portInfo.toClientType)))
          return failure();
      if (portInfo.toServerType)
        if (failed(storeType(portInfo.toServerType)))
          return failure();
    }
  }
  return success();
}

LogicalResult CPPAPI::emit() {
  // Walk and collect the type data.
  if (failed(gatherTypes()))
    return failure();

  os << "#pragma once\n\n";

  os << "// The ESI C++ API relies on the refl-cpp library for type "
        "introspection. This must be provided by the user.\n";
  os << "// See https://github.com/veselink1/refl-cpp \n";
  os << "#include \"refl.hpp\"\n\n";

  os << "#include <cstdint>\n";
  os << "#include \"esi/backends/capnp.h\"\n";
  os << "\n// Include the generated Cap'nProto schema header. This must "
        "defined by the build system.\n";
  os << "#include ESI_COSIM_CAPNP_H\n";
  os << "\n\n";

  os << "namespace esi {\n";
  os << "namespace runtime {\n\n";

  if (failed(emitTypes()) || failed(emitServiceDeclarations()) ||
      failed(emitDesignModules()))
    return failure();

  os << "} // namespace runtime\n";
  os << "} // namespace esi\n\n";

  os << "// ESI dynamic reflection support\n";
  if (failed(emitGlobalNamespace()))
    return failure();

  return success();
}

LogicalResult CPPAPI::emitServiceDeclarations() {
  // Locate all of the service declarations which are needed by the
  // services in the service hierarchy.
  for (auto serviceDeclOp : module.getOps<ServiceDeclOpInterface>()) {
    auto cppService = CPPService(serviceDeclOp, types);
    if (failed(cppService.write(os)))
      return failure();
    cppServices.push_back(cppService);
  }

  return success();
}

LogicalResult CPPAPI::emitGlobalNamespace() {
  // Emit ESI type reflection classes.
  llvm::SmallVector<std::string> namespaces = {"esi", "runtime", "ESITypes"};
  for (auto &cppType : types)
    cppType.second.writeReflection(os, namespaces);

  return success();
}

LogicalResult CPPAPI::emitTypes() {
  os << "class ESITypes {\n";
  os << "public:\n";
  os.indent();

  // Iterate through the various types and emit their CPP APIs.
  for (const auto &cppType : types) {
    if (failed(cppType.second.write(os)))
      // If we fail during an emission, dump out early since the output may
      // be corrupted.
      return failure();
  }

  os.unindent();
  os << "};\n\n";

  return success();
}

LogicalResult CPPAPI::emitDesignModules() {
  // Get a list of metadata ops which originated in modules (path is empty).
  SmallVector<
      std::pair<hw::HWModuleLike, SmallVector<ServiceHierarchyMetadataOp, 0>>>
      modsWithLocalServices;
  for (auto hwmod : module.getOps<hw::HWModuleLike>()) {
    SmallVector<ServiceHierarchyMetadataOp, 0> metadataOps;
    hwmod.walk([&metadataOps](ServiceHierarchyMetadataOp md) {
      if (md.getServerNamePath().empty() && md.getImplType() == "cosim")
        metadataOps.push_back(md);
    });
    if (!metadataOps.empty())
      modsWithLocalServices.push_back(std::make_pair(hwmod, metadataOps));
  }

  SmallVector<CPPDesignModule> designMods;
  for (auto &mod : modsWithLocalServices)
    designMods.push_back(CPPDesignModule(mod.first, mod.second, cppServices));

  // Write modules
  for (auto &designMod : designMods) {
    if (failed(designMod.write(os)))
      return failure();
    os << "\n";
  }
  return success();
}

LogicalResult circt::esi::cppapi::exportCPPAPI(ModuleOp module,
                                               llvm::raw_ostream &os) {
  CPPAPI api(module, os);
  return api.emit();
}
