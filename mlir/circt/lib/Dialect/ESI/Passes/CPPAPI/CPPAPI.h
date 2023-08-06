//===- CPPAPI.h - ESI C++ api -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Code for generating the ESI C++ API.
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef CIRCT_DIALECT_ESI_PASSES_CPPAPI_CPPAPI_H
#define CIRCT_DIALECT_ESI_PASSES_CPPAPI_CPPAPI_H

#include "circt/Dialect/ESI/APIUtilities.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/MapVector.h"

#include <memory>

namespace circt {
namespace esi {
namespace cppapi {

// Writes the C++ API for the given module to the provided output stream.
LogicalResult exportCPPAPI(ModuleOp module, llvm::raw_ostream &os);

// Generate and reason about a C++ type for a particular Cap'nProto and MLIR
// type.
class CPPType : public ESIAPIType {
public:
  using ESIAPIType::ESIAPIType;

  /// Returns true if the type is supported for the CPP API.
  bool isSupported() const override;

  /// Write out the C++ name of this type.
  void writeCppName(llvm::raw_ostream &os) const;

  /// Write out the type in its entirety.
  mlir::LogicalResult write(mlir::raw_indented_ostream &os) const;

  // Emits an RTTR registration for this type. If provided, the `namespace`
  // should indicate the namespace wherein this type was emitted.
  void writeReflection(mlir::raw_indented_ostream &os,
                       llvm::ArrayRef<std::string> namespaces) const;
};

struct CPPEndpoint {
  CPPEndpoint(esi::ServicePortInfo portInfo,
              const llvm::MapVector<mlir::Type, CPPType> &types)
      : portInfo(portInfo), types(types) {}
  StringRef getName() const { return portInfo.name.getValue(); }
  std::string getTypeName() const { return "T" + getName().str(); }
  std::string getPointerTypeName() const { return getTypeName() + "Ptr"; }
  LogicalResult writeType(Location loc, mlir::raw_indented_ostream &os) const;
  LogicalResult writeDecl(Location loc, mlir::raw_indented_ostream &os) const;

  esi::ServicePortInfo portInfo;

  // A mapping of MLIR types to their CPPType counterparts. Ensures
  // consistency between the emitted type signatures and those used in the
  // service endpoint API.
  const llvm::MapVector<mlir::Type, CPPType> &types;
};

class CPPService {
public:
  CPPService(esi::ServiceDeclOpInterface service,
             const llvm::MapVector<mlir::Type, CPPType> &types);

  // Return the name of this service.
  StringRef name() const {
    return SymbolTable::getSymbolName(service).getValue();
  }

  // Write out the C++ API of this service.
  LogicalResult write(mlir::raw_indented_ostream &os);

  esi::ServiceDeclOpInterface getService() { return service; }
  llvm::SmallVector<ServicePortInfo> getPorts();
  CPPEndpoint *getPort(llvm::StringRef portName);

  auto &getEndpoints() { return endpoints; }

private:
  esi::ServiceDeclOpInterface service;

  // Note: cannot use llvm::SmallVector on a forward declared class.
  llvm::SmallVector<std::shared_ptr<CPPEndpoint>> endpoints;
};

class CPPDesignModule {
public:
  CPPDesignModule(hw::HWModuleLike mod,
                  SmallVectorImpl<ServiceHierarchyMetadataOp> &services,
                  llvm::SmallVectorImpl<CPPService> &cppServices)
      : mod(mod), services(services), cppServices(cppServices) {}

  llvm::StringRef name() { return mod.getModuleName(); }
  LogicalResult write(mlir::raw_indented_ostream &ios);

private:
  hw::HWModuleLike mod;
  SmallVectorImpl<ServiceHierarchyMetadataOp> &services;
  SmallVectorImpl<CPPService> &cppServices;
};

} // namespace cppapi
} // namespace esi
} // namespace circt

#endif // CIRCT_DIALECT_ESI_PASSES_CPPAPI_CPPAPI_H
