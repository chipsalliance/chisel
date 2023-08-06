//===- esi.h - ESI system C++ API -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ESI system C++ API which all backends must implement.
//
// DO NOT EDIT!
// This file is distributed as part of an ESI package. The source for this file
// should always be modified within CIRCT.
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef ESI_RUNTIME_ESI_H
#define ESI_RUNTIME_ESI_H

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace esi {
namespace runtime {

/*
// A backend is anything that implements all of the TBackend:: types and
functions
// used in this file, e.g.:

class TBackend {
public:
  template <typename WriteType, typename ReadType>
  using ReadWritePort = ...

  template <typename WriteType>
  using WritePort = ...

  template <typename ReadType>
  using ReadPort = ...
};
*/

// Base class for all ports.
template <typename TBackend>
class Port {

public:
  Port(const std::vector<std::string> &clientPath, TBackend &backend,
       const std::string &implType)
      : backend(&backend), clientPath(clientPath) {}

  // Initializes this port - have to do it post-construction due to initBackend
  // being pure virtual.
  void init() { initBackend(); }

  // Hook for the backend port implementation to initialize the port.
  virtual void initBackend() = 0;

protected:
  TBackend *backend = nullptr;
  std::vector<std::string> clientPath;
};

template <typename WriteType, typename ReadType, typename TBackend>
class ReadWritePort
    : public TBackend::template ReadWritePort<WriteType, ReadType> {
public:
  using Impl = typename TBackend::template ReadWritePort<WriteType, ReadType>;
  static_assert(std::is_base_of<runtime::Port<TBackend>, Impl>::value,
                "Backend port must be a subclass of runtime::Port");

  auto operator()(WriteType arg) { return getImpl()->operator()(arg); }
  Impl *getImpl() { return static_cast<Impl *>(this); }

  ReadWritePort(const std::vector<std::string> &clientPath, TBackend &backend,
                const std::string &implType)
      : Impl(clientPath, backend, implType) {}
};

template <typename WriteType, typename TBackend>
class WritePort : public TBackend::template WritePort<WriteType> {
public:
  using Impl = typename TBackend::template WritePort<WriteType>;
  static_assert(std::is_base_of<runtime::Port<TBackend>, Impl>::value,
                "Backend port must be a subclass of runtime::Port");

  auto operator()(WriteType arg) { return getImpl()->operator()(arg); }
  Impl *getImpl() { return static_cast<Impl *>(this); }

  WritePort(const std::vector<std::string> &clientPath, TBackend &backend,
            const std::string &implType)
      : Impl(clientPath, backend, implType) {}
};

template <typename ReadType, typename TBackend>
class ReadPort : public TBackend::template ReadPort<ReadType> {
public:
  using Impl = typename TBackend::template ReadPort<ReadType>;
  static_assert(std::is_base_of<runtime::Port<TBackend>, Impl>::value,
                "Backend port must be a subclass of runtime::Port");

  auto operator()() { return getImpl()->operator()(); }
  Impl *getImpl() { return static_cast<Impl *>(this); }

  ReadPort(const std::vector<std::string> &clientPath, TBackend &backend,
           const std::string &implType)
      : Impl(clientPath, backend, implType) {}
};

template <typename TBackend>
class Module {
public:
  Module(const std::vector<std::shared_ptr<Port<TBackend>>> &ports)
      : ports(ports) {}

  // Initializes this module
  void init() {
    // Initialize all ports
    for (auto &port : ports)
      port->init();
  }

protected:
  // A handle to all ports in this module.
  std::vector<std::shared_ptr<Port<TBackend>>> ports;
};

} // namespace runtime
} // namespace esi

#endif // ESI_RUNTIME_ESI_H
