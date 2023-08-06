//===- capnp.h - ESI C++ cosimulation Cap'n'proto backend -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a specialization of the ESI C++ API for the Cap'n'proto cosimulation
// backend.
//
// DO NOT EDIT!
// This file is distributed as part of an ESI package. The source for this file
// should always be modified within CIRCT (lib/dialect/ESI/runtime/cpp/esi.h).
//
//===----------------------------------------------------------------------===//

#pragma once

#include "esi/esi.h"
#include "refl.hpp"

#include <capnp/dynamic.h>
#include <capnp/ez-rpc.h>
#include <capnp/message.h>
#include <capnp/schema.h>

// Assert that an ESI_COSIM_CAPNP_H variable is defined. This is the capnp
// header file generated from the ESI schema, containing definitions for e.g.
// CosimDpiServer, ...
#ifndef ESI_COSIM_CAPNP_H
#error "ESI_COSIM_CAPNP_H must be defined to include this file"
#endif
#include ESI_COSIM_CAPNP_H

using namespace capnp;

namespace esi {
namespace runtime {
namespace cosim {

// Uses the capnp dynamic interface in conjunction with ESI type reflection to
// de-serialize a capnp message into an ESI value.
template <typename TESIType>
TESIType fromCapnp(DynamicValue::Reader value) {
  TESIType ret;
  if (value.getType() != DynamicValue::STRUCT)
    throw std::runtime_error(
        "Expected a struct value (all ESI types should be capnp structs)");

  auto structValue = value.as<DynamicStruct>();
  for_each(refl::reflect(ret).members, [&](auto member) {
    auto fieldValue = structValue.get(member.name);
    using ValueType = typename decltype(member)::value_type;
    auto castValue = fieldValue.template as<ValueType>();
    member(ret) = castValue;
  });

  return ret;
}

// A function like above, but which uses Capnproto dynamic message building
// together with the refl-cpp reflection to constuct the Capnp message.
template <typename TESIType>
void toCapnp(TESIType &value, DynamicValue::Builder &message) {
  if (message.getType() != DynamicValue::STRUCT)
    throw std::runtime_error(
        "Expected a struct value (all ESI types should be capnp structs)");

  auto structMessage = message.as<DynamicStruct>();
  for_each(refl::reflect(value).members, [&](auto member) {
    auto &fieldValue = member(value);
    structMessage.set(member.name, fieldValue);
  });
}

namespace detail {
// Custom type to hold the interface descriptions because i can't for the life
// of me figure out how to cleanly keep capnproto messages around...
struct EsiDpiInterfaceDesc {
  std::string endpointID;
  uint64_t sendTypeID;
  uint64_t recvTypeID;
};
} // namespace detail

using EsiDpiInterfaceDesc = detail::EsiDpiInterfaceDesc;

template <typename WriteType, typename ReadType>
class CapnpReadWritePort;

template <typename WriteType>
class CapnpWritePort;

template <typename ReadType>
class CapnpReadPort;

class CapnpBackend {
public:
  // Using directives to point the base class implementations to the cosim
  // port implementations.

  template <typename WriteType, typename ReadType>
  using ReadWritePort = CapnpReadWritePort<WriteType, ReadType>;

  template <typename WriteType>
  using WritePort = CapnpWritePort<WriteType>;

  template <typename ReadType>
  using ReadPort = CapnpReadPort<ReadType>;

  CapnpBackend(const std::string &host, uint64_t hostPort) {
    ezClient = std::make_unique<capnp::EzRpcClient>(host, hostPort);
    dpiClient = std::make_unique<CosimDpiServer::Client>(
        ezClient->getMain<CosimDpiServer>());

    list();
  }

  // Returns a list of all available endpoints.
  const std::vector<detail::EsiDpiInterfaceDesc> &list() {
    if (endpoints.has_value())
      return *endpoints;

    // Query the DPI server for a list of available endpoints.
    auto listReq = dpiClient->listRequest();
    auto ifaces = listReq.send().wait(ezClient->getWaitScope()).getIfaces();
    endpoints = std::vector<detail::EsiDpiInterfaceDesc>();
    for (auto iface : ifaces) {
      detail::EsiDpiInterfaceDesc desc;
      desc.endpointID = iface.getEndpointID().cStr();
      desc.sendTypeID = iface.getSendTypeID();
      desc.recvTypeID = iface.getRecvTypeID();
      endpoints->push_back(desc);
    }

    // print out the endpoints
    for (auto ep : *endpoints) {
      std::cout << "Endpoint: " << ep.endpointID << std::endl;
      std::cout << "  Send Type: " << ep.sendTypeID << std::endl;
      std::cout << "  Recv Type: " << ep.recvTypeID << std::endl;
    }

    return *endpoints;
  }

  template <typename CnPWriteType, typename CnPReadType>
  auto getPort(const std::vector<std::string> &clientPath) {
    // Join client path into a single string with '.' as a separator.
    std::string clientPathStr;
    for (auto &path : clientPath) {
      if (!clientPathStr.empty())
        clientPathStr += '_';
      clientPathStr += path;
    }

    // Everything is nested under "TOP.top"
    clientPathStr = "TOP.top." + clientPathStr;

    auto openReq = dpiClient->openRequest<CnPWriteType, CnPReadType>();

    // Scan through the available endpoints to find the requested one.
    bool found = false;
    for (auto &ep : list()) {
      auto epid = ep.endpointID;
      if (epid == clientPathStr) {
        auto iface = openReq.getIface();
        iface.setEndpointID(epid);
        iface.setSendTypeID(ep.sendTypeID);
        iface.setRecvTypeID(ep.recvTypeID);
        found = true;
        break;
      }
    }

    if (!found)
      throw std::runtime_error("Could not find endpoint: " + clientPathStr);

    // Open the endpoint.
    auto openResp = openReq.send().wait(ezClient->getWaitScope());
    return openResp.getIface();
  }

  bool supportsImpl(const std::string &implType) {
    // The cosim backend only supports cosim connectivity implementations
    return implType == "cosim";
  }

  kj::WaitScope &getWaitScope() { return ezClient->getWaitScope(); }

protected:
  std::unique_ptr<capnp::EzRpcClient> ezClient;
  std::unique_ptr<CosimDpiServer::Client> dpiClient;
  std::optional<std::vector<detail::EsiDpiInterfaceDesc>> endpoints;
};

template <typename WriteType, typename ReadType>
class CapnpReadWritePort : public Port<CapnpBackend> {
  using BasePort = Port<CapnpBackend>;

public:
  CapnpReadWritePort(const std::vector<std::string> &clientPath,
                     CapnpBackend &backend, const std::string &implType)
      : BasePort(clientPath, backend, implType) {}

  ReadType operator()(WriteType arg) {
    auto req = port->sendRequest();
    MallocMessageBuilder mmb;
    auto dynBuilder = DynamicValue::Builder(req.getMsg());
    toCapnp<WriteType>(arg, dynBuilder);
    req.send().wait(this->backend->getWaitScope());
    std::optional<capnp::Response<typename EsiDpiEndpoint<
        typename WriteType::CPType, typename ReadType::CPType>::RecvResults>>
        resp;
    do {
      auto recvReq = port->recvRequest();
      recvReq.setBlock(false);

      resp = recvReq.send().wait(this->backend->getWaitScope());
    } while (!resp->getHasData());
    auto data = resp->getResp();
    return fromCapnp<ReadType>(data);
  }

  void initBackend() override {
    port =
        backend->getPort<typename WriteType::CPType, typename ReadType::CPType>(
            clientPath);
  }

private:
  // Handle to the underlying endpoint.
  std::optional<typename ::EsiDpiEndpoint<typename WriteType::CPType,
                                          typename ReadType::CPType>::Client>
      port;
};

template <typename WriteType>
class CapnpWritePort : public Port<CapnpBackend> {
  using BasePort = Port<CapnpBackend>;

public:
  CapnpWritePort(const std::vector<std::string> &clientPath,
                 CapnpBackend &backend, const std::string &implType)
      : BasePort(clientPath, backend, implType) {}

  void initBackend() override {
    port = backend->getPort<typename WriteType::CPType, ::I1>(clientPath);
  }

  void operator()(WriteType arg) {
    auto req = port->sendRequest();
    auto dynBuilder = DynamicValue::Builder(req.getMsg());
    toCapnp<WriteType>(arg, dynBuilder);
    req.send().wait(this->backend->getWaitScope());
  }

private:
  // Handle to the underlying endpoint.
  std::optional<
      typename ::EsiDpiEndpoint<typename WriteType::CPType, ::I1>::Client>
      port;
};

template <typename ReadType>
class CapnpReadPort : public Port<CapnpBackend> {
  using BasePort = Port<CapnpBackend>;

public:
  CapnpReadPort(const std::vector<std::string> &clientPath,
                CapnpBackend &backend, const std::string &implType)
      : BasePort(clientPath, backend, implType) {}

  void initBackend() override {
    port = backend->getPort<::I1, typename ReadType::CPType>(clientPath);
  }

  ReadType operator()() {
    std::optional<capnp::Response<
        typename EsiDpiEndpoint<::I1, typename ReadType::CPType>::RecvResults>>
        resp;
    do {
      auto recvReq = port->recvRequest();
      recvReq.setBlock(false);

      resp = recvReq.send().wait(this->backend->getWaitScope());
    } while (!resp->getHasData());
    auto data = resp->getResp();
    return fromCapnp<ReadType>(data);
  }

private:
  // Handle to the underlying endpoint.
  std::optional<
      typename ::EsiDpiEndpoint<::I1, typename ReadType::CPType>::Client>
      port;
};

} // namespace cosim
} // namespace runtime
} // namespace esi
