//===- CPPAPI.cpp -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Code for generating the ESI C++ API.
//
//===----------------------------------------------------------------------===//

#include "CPPAPI.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/ESI/ESITypes.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/SV/SVOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Format.h"

#include <initializer_list>
#include <string>

using namespace circt;
using namespace esi;
using namespace cppapi;

//===----------------------------------------------------------------------===//
// CPPType class implementation.
//===----------------------------------------------------------------------===//

static std::string lowercase(llvm::StringRef str) {
  std::string s;
  llvm::raw_string_ostream ss(s);
  for (auto &c : str)
    ss << llvm::toLower(c);
  return s;
}

static std::string joinStringAttrArray(mlir::ArrayAttr strings,
                                       llvm::StringRef delimiter) {
  std::string s;
  llvm::raw_string_ostream ss(s);
  llvm::interleave(
      strings.getAsValueRange<StringAttr>(), ss, [&](auto str) { ss << str; },
      delimiter);
  return s;
}

static void emitCPPType(Type type, mlir::raw_indented_ostream &os) {
  llvm::TypeSwitch<Type>(type)
      .Case([&os](IntegerType intTy) {
        auto w = intTy.getWidth();
        if (w == 0) {
          os << "void";
        } else if (w == 1) {
          os << "bool";
        } else {
          if (intTy.isSigned())
            os << "int";
          else
            os << "uint";

          // Round up.
          if (w <= 8)
            os << "8";
          else if (w <= 16)
            os << "16";
          else if (w <= 32)
            os << "32";
          else if (w <= 64)
            os << "64";
          else
            assert(false && "Type not supported. Integer too wide. Please "
                            "check support first with isSupported()");

          os << "_t";
        }
      })
      .Case<hw::ArrayType, hw::StructType>([](auto) {
        assert(false && "Structs containing List and Structs not supported");
      })
      .Default([](Type) {
        assert(false && "Type not supported. Please check support first with "
                        "isSupported()");
      });
}

static bool isZeroWidthInt(Type type) {
  return type.isa<NoneType>() ||
         (type.isa<IntegerType>() && type.cast<IntegerType>().getWidth() == 0);
}

LogicalResult CPPType::write(mlir::raw_indented_ostream &os) const {

  os << "struct " << name() << " {\n";
  os.indent();

  os << "// Data members:\n";
  auto fieldTypes = getFields();
  for (auto field : fieldTypes) {

    // Do not emit zero-width fields.
    if (isZeroWidthInt(field.type))
      os << "// Zero-width field: ";

    emitCPPType(field.type, os);
    os << " " << field.name.getValue() << ";";
    // Specify the mlir type.
    os << "\t// MLIR type is " << field.type << "\n";
  }
  os << "\n";

  bool isI0 = false;
  if (fieldTypes.size() == 1) {
    os << "// Unary types have convenience constructors\n";
    os << name() << "() = default;\n";

    auto fieldType = fieldTypes[0].type;
    if (!isZeroWidthInt(fieldType)) {

      os << name() << "(";
      auto fieldName = fieldTypes[0].name.getValue();
      emitCPPType(fieldType, os);
      os << " " << fieldName << ") : " << fieldName << "(" << fieldName << ")"
         << " {}\n\n";

      os << "// Unary types have convenience conversion operators\n";
      os << "operator ";
      emitCPPType(fieldType, os);
      os << "() const { return " << fieldName << "; }\n\n";
    } else {
      isI0 = true;
    }
  }

  // Comparison operator
  os << "// Equality operator\n";
  os << "// TODO (c++20): emit default <=> operator\n";
  os << "auto operator==(const " << name() << " &other) const {\n";
  os.indent();
  os << "return ";
  llvm::interleave(
      fieldTypes, os,
      [&](auto field) {
        os << "(";
        if (isZeroWidthInt(field.type))
          os << "/*i0==*/true";
        else
          os << field.name.getValue() << " == other." << field.name.getValue();
        os << ")";
      },
      " && ");
  os << ";\n";
  os.unindent();
  os << "}\n\n";

  // != operator
  os << "// Inequality operator\n";
  os << "auto operator!=(const " << name() << " &other) const {\n";
  os.indent();
  os << "return !(*this == other);\n";
  os.unindent();
  os << "}\n\n";

  // Stream operator
  os << "// Stream operator\n";
  os << "friend std::ostream &operator<<(std::ostream &os, const " << name()
     << " &val) {\n";
  os.indent();
  os << "os << \"" << name() << "(\";\n";
  for (auto [idx, field] : llvm::enumerate(fieldTypes)) {
    if (isZeroWidthInt(field.type))
      continue;
    os << "os << \"" << field.name.getValue() << ": \";\n";

    // A bit of a hack - (u)int8_t types will by default be printed as chars.
    // We want to avoid this and just print the underlying value.
    os << "os << ";
    if (field.type.getIntOrFloatBitWidth() > 1 &&
        field.type.getIntOrFloatBitWidth() <= 8)
      os << "(uint32_t)";
    os << "val." << field.name.getValue() << ";\n";
    if (idx != fieldTypes.size() - 1)
      os << "os << \", \";\n";
  }
  os << "os << \")\";\n";
  os << "return os;\n";
  os.unindent();
  os << "}\n\n";

  // Capnproto type (todo: remove)
  os << "// Cap'nProto type which this ESI type maps to\n";
  os << "using CPType = ::";
  if (isI0)
    os << "UntypedData";
  else
    os << name();

  os << ";\n";

  os.unindent();
  os << "};\n\n";

  return success();
}

bool CPPType::isSupported() const {
  // C++ doesn't support esi.any (yet)
  return !type.isa<esi::AnyType>() && ESIAPIType::isSupported();
}

void CPPType::writeReflection(mlir::raw_indented_ostream &os,
                              llvm::ArrayRef<std::string> namespaces) const {
  os << "REFL_AUTO (\n";
  os.indent();

  std::string ns;
  if (!namespaces.empty()) {
    for (auto &n : namespaces)
      ns += n + "::";
  }

  os << "type(" << ns << name() << ")";

  for (auto &field : getFields()) {
    os << "\n";
    bool isI0 = field.type.getIntOrFloatBitWidth() == 0;
    if (isI0)
      os << "//";
    os << ", field(" << field.name.getValue() << ")";
  };

  os.unindent();
  os << "\n)\n";
}

//===----------------------------------------------------------------------===//
// Endpoint class implementation.
//===----------------------------------------------------------------------===//

LogicalResult CPPEndpoint::writeType(Location loc,
                                     mlir::raw_indented_ostream &os) const {
  auto emitType = [&](llvm::StringRef dir, mlir::Type type) -> LogicalResult {
    type = esi::innerType(type);
    auto cppTypeIt = types.find(type);
    if (cppTypeIt == types.end()) {

      emitError(loc) << "Type " << type
                     << " not found in set of CPP API types!";
      return failure();
    }

    os << "/*" << dir << "Type=*/ ESITypes::" << cppTypeIt->second.name();
    return success();
  };

  if (portInfo.toClientType && portInfo.toServerType) {
    os << "esi::runtime::ReadWritePort<";
    if (failed(emitType("read", portInfo.toClientType)))
      return failure();
    os << ", ";
    if (failed(emitType("write", portInfo.toServerType)))
      return failure();
  } else if (portInfo.toServerType) {
    os << "esi::runtime::ReadPort<";
    if (failed(emitType("read", portInfo.toServerType)))
      return failure();
  } else if (portInfo.toClientType) {
    os << "esi::runtime::WritePort<";
    if (failed(emitType("write", portInfo.toClientType)))
      return failure();
  } else {
    llvm_unreachable("Port has no type");
  }
  os << ", TBackend>";
  return success();
}

LogicalResult CPPEndpoint::writeDecl(Location loc,
                                     mlir::raw_indented_ostream &os) const {
  os << "using " << getTypeName() << " = ";
  if (failed(writeType(loc, os)))
    return failure();
  os << ";\n";
  os << "using " << getPointerTypeName() << " = std::shared_ptr<"
     << getTypeName() << ">;\n\n";
  return success();
}

LogicalResult CPPService::write(mlir::raw_indented_ostream &os) {
  auto loc = service.getLoc();
  os << "template <typename TBackend>\n";
  os << "class " << name() << " : public esi::runtime::Module<TBackend> {\n";
  os.indent();
  os << "  using Port = esi::runtime::Port<TBackend>;\n"
     << "public:\n";
  os << "// Port type declarations.\n";

  for (auto &ep : endpoints)
    if (failed(ep->writeDecl(loc, os)))
      return failure();

  os << "// Constructor initializes each endpoint.\n";
  os << name() << "(";
  llvm::interleaveComma(endpoints, os, [&](auto &ep) {
    os << ep->getPointerTypeName() << " " << ep->getName();
  });
  os << ")\n";
  os << "// Initialize base class with knowledge of all endpoints.\n";
  os << ": esi::runtime::Module<TBackend>({";
  llvm::interleaveComma(endpoints, os, [&](auto &ep) { os << ep->getName(); });
  os << "}),\n";

  os << "// Initialize each individual endpoint handle.\n";
  llvm::interleaveComma(endpoints, os, [&](auto &ep) {
    os << ep->getName() << "(" << ep->getName() << ")";
  });
  os << " {}\n\n";

  os << "// Endpoint handles\n";
  for (auto &ep : endpoints)
    os << ep->getPointerTypeName() << " " << ep->getName() << ";\n";

  os.unindent();
  os << "};\n\n";
  return success();
}

CPPService::CPPService(esi::ServiceDeclOpInterface service,
                       const llvm::MapVector<mlir::Type, CPPType> &types)
    : service(service) {
  llvm::SmallVector<esi::ServicePortInfo> portList;
  service.getPortList(portList);
  for (auto &portInfo : portList)
    endpoints.push_back(std::make_shared<CPPEndpoint>(portInfo, types));
}

llvm::SmallVector<esi::ServicePortInfo> CPPService::getPorts() {
  llvm::SmallVector<ServicePortInfo> ports;
  getService().getPortList(ports);
  return ports;
}

CPPEndpoint *CPPService::getPort(llvm::StringRef portName) {
  auto *it = llvm::find_if(
      endpoints, [&](auto &ep) { return ep->portInfo.name == portName; });

  if (it == endpoints.end())
    return nullptr;

  return it->get();
}

LogicalResult CPPDesignModule::write(mlir::raw_indented_ostream &ios) {
  ios << "template <typename TBackend>\n";
  ios << "class " << name() << " {\n";
  ios << "public:\n";
  ios.indent();

  // Locate the inner services.
  struct ServiceInfo {
    std::string cppMemberName;
    ArrayAttr clients;
  };
  llvm::MapVector<CPPService *, ServiceInfo> innerServices;
  for (auto &service : services) {
    // lookup service in CPPServices
    auto serviceSym = *service.getServiceSymbol();
    auto *cppServiceIt = llvm::find_if(cppServices, [&](auto &cppService) {
      auto sym = SymbolTable::getSymbolName(cppService.getService());
      return sym == serviceSym;
    });
    if (cppServiceIt == cppServices.end())
      return mod.emitOpError()
             << "Service " << serviceSym << " not found in CPPServices.";

    ServiceInfo info;
    info.cppMemberName = lowercase(cppServiceIt->name().str());
    info.clients = service.getClients();
    ios << "std::unique_ptr<" << cppServiceIt->name() << "<TBackend>> "
        << info.cppMemberName << ";\n";
    innerServices[cppServiceIt] = info;
  }

  // Add constructor
  constexpr static std::string_view backendName = "backend";
  ios << name() << "(TBackend& " << backendName << ") {\n";
  ios.indent();
  for (auto &[innerCppService, serviceInfo] : innerServices) {
    ios << "// " << innerCppService->name() << " initialization.\n";

    struct ClientPortName {
      std::string servicePortName;
      std::string clientPortName;
    };
    llvm::SmallVector<ClientPortName> portNames;

    for (auto [i, client] : llvm::enumerate(serviceInfo.clients)) {
      // cast client to a dictionary attribute
      auto clientDict = client.cast<DictionaryAttr>();
      auto clientName = clientDict.getAs<ArrayAttr>("client_name");
      auto clientNameRange = clientName.getAsValueRange<StringAttr>();
      auto port = clientDict.getAs<hw::InnerRefAttr>("port");

      auto portName = joinStringAttrArray(clientName, "_");
      auto *ep = innerCppService->getPort(port.getName());
      ios << "auto " << portName << " = std::make_shared<";
      portNames.push_back(ClientPortName{port.getName().str(), portName});
      if (failed(ep->writeType(mod.getLoc(), ios)))
        return failure();
      ios << ">(std::vector<std::string>{";
      llvm::interleaveComma(clientNameRange, ios, [&](auto clientHierName) {
        ios << "\"" << clientHierName.str() << "\"";
      });
      ios << "}, " << backendName << ", \"cosim\");\n";
    }

    // And instantiate the member using the instantiated ports. Make sure that
    // they are in the same order as the endpoints in the service to match the
    // order of the ports in the constructor.
    auto servicePorts = innerCppService->getPorts();
    llvm::sort(portNames, [&](auto &a, auto &b) {
      auto aIt = llvm::find_if(servicePorts, [&](auto &port) {
        return port.name == a.servicePortName;
      });
      auto bIt = llvm::find_if(servicePorts, [&](auto &port) {
        return port.name == b.servicePortName;
      });
      auto aIdx = std::distance(servicePorts.begin(), aIt);
      auto bIdx = std::distance(servicePorts.begin(), bIt);
      return aIdx < bIdx;
    });

    ios << serviceInfo.cppMemberName << " = std::make_unique<"
        << innerCppService->name() << "<TBackend>>(";
    llvm::interleaveComma(portNames, ios, [&](auto &portName) {
      ios << portName.clientPortName;
    });
    ios << ");\n";
    // and initialize...
    ios << serviceInfo.cppMemberName << "->init();\n";
  }

  ios.unindent();
  ios << "}\n";
  ios.unindent();
  ios << "};\n";
  return success();
}
