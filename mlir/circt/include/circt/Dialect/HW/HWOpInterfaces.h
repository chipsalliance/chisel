//===- HWOpInterfaces.h - Declare HW op interfaces --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation interfaces for the HW dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_HWOPINTERFACES_H
#define CIRCT_DIALECT_HW_HWOPINTERFACES_H

#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/HW/InnerSymbolTable.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"

namespace circt {
namespace hw {

/// This holds the name, type, direction of a module's ports
struct PortInfo : public ModulePort {
  /// This is the argument index or the result index depending on the direction.
  /// "0" for an output means the first output, "0" for a in/inout means the
  /// first argument.
  size_t argNum = ~0U;

  /// The optional symbol for this port.
  InnerSymAttr sym = {};
  DictionaryAttr attrs = {};
  LocationAttr loc = {};

  StringRef getName() const { return name.getValue(); }
  bool isInput() const { return dir == ModulePort::Direction::Input; }
  bool isOutput() const { return dir == ModulePort::Direction::Output; }
  bool isInOut() const { return dir == ModulePort::Direction::InOut; }

  /// Return a unique numeric identifier for this port.
  ssize_t getId() const { return isOutput() ? argNum : (-1 - argNum); };
};

/// This holds a decoded list of input/inout and output ports for a module or
/// instance.
struct ModulePortInfo {
  explicit ModulePortInfo(ArrayRef<PortInfo> inputs,
                          ArrayRef<PortInfo> outputs) {
    ports.insert(ports.end(), inputs.begin(), inputs.end());
    ports.insert(ports.end(), outputs.begin(), outputs.end());
  }

  explicit ModulePortInfo(ArrayRef<PortInfo> mergedPorts)
      : ports(mergedPorts.begin(), mergedPorts.end()) {}

  using iterator = SmallVector<PortInfo>::iterator;
  using const_iterator = SmallVector<PortInfo>::const_iterator;

  iterator begin() { return ports.begin(); }
  iterator end() { return ports.end(); }
  const_iterator begin() const { return ports.begin(); }
  const_iterator end() const { return ports.end(); }

  using PortDirectionRange = llvm::iterator_range<
      llvm::filter_iterator<iterator, std::function<bool(const PortInfo &)>>>;

  using ConstPortDirectionRange = llvm::iterator_range<llvm::filter_iterator<
      const_iterator, std::function<bool(const PortInfo &)>>>;

  PortDirectionRange getPortsOfDirection(bool input) {
    std::function<bool(const PortInfo &)> predicateFn;
    if (input) {
      predicateFn = [](const PortInfo &port) -> bool {
        return port.dir == ModulePort::Direction::Input ||
               port.dir == ModulePort::Direction::InOut;
      };
    } else {
      predicateFn = [](const PortInfo &port) -> bool {
        return port.dir == ModulePort::Direction::Output;
      };
    }
    return llvm::make_filter_range(ports, predicateFn);
  }

  ConstPortDirectionRange getPortsOfDirection(bool input) const {
    std::function<bool(const PortInfo &)> predicateFn;
    if (input) {
      predicateFn = [](const PortInfo &port) -> bool {
        return port.dir == ModulePort::Direction::Input ||
               port.dir == ModulePort::Direction::InOut;
      };
    } else {
      predicateFn = [](const PortInfo &port) -> bool {
        return port.dir == ModulePort::Direction::Output;
      };
    }
    return llvm::make_filter_range(ports, predicateFn);
  }

  PortDirectionRange getInputs() { return getPortsOfDirection(true); }

  PortDirectionRange getOutputs() { return getPortsOfDirection(false); }

  ConstPortDirectionRange getInputs() const {
    return getPortsOfDirection(true);
  }

  ConstPortDirectionRange getOutputs() const {
    return getPortsOfDirection(false);
  }

  size_t size() const { return ports.size(); }
  size_t sizeInputs() const {
    auto r = getInputs();
    return std::distance(r.begin(), r.end());
  }
  size_t sizeOutputs() const {
    auto r = getOutputs();
    return std::distance(r.begin(), r.end());
  }

  size_t portNumForInput(size_t idx) const {
    size_t port = 0;
    while (idx || ports[port].isOutput()) {
      if (!ports[port].isOutput())
        --idx;
      ++port;
    }
    return port;
  }

  size_t portNumForOutput(size_t idx) const {
    size_t port = 0;
    while (idx || !ports[port].isOutput()) {
      if (ports[port].isOutput())
        --idx;
      ++port;
    }
    return port;
  }

  PortInfo &at(size_t idx) { return ports[idx]; }
  PortInfo &atInput(size_t idx) { return ports[portNumForInput(idx)]; }
  PortInfo &atOutput(size_t idx) { return ports[portNumForOutput(idx)]; }

  const PortInfo &at(size_t idx) const { return ports[idx]; }
  const PortInfo &atInput(size_t idx) const {
    return ports[portNumForInput(idx)];
  }
  const PortInfo &atOutput(size_t idx) const {
    return ports[portNumForOutput(idx)];
  }

  void eraseInput(size_t idx) {
    assert(idx < sizeInputs());
    ports.erase(ports.begin() + portNumForInput(idx));
  }

private:
  /// This contains a list of all ports.  Input first.
  SmallVector<PortInfo> ports;
};

// This provides capability for looking up port indices based on port names.
struct ModulePortLookupInfo {
  FailureOr<unsigned>
  lookupPortIndex(const llvm::DenseMap<StringAttr, unsigned> &portMap,
                  StringAttr name) const {
    auto it = portMap.find(name);
    if (it == portMap.end())
      return failure();
    return it->second;
  }

public:
  explicit ModulePortLookupInfo(MLIRContext *ctx,
                                const ModulePortInfo &portInfo)
      : ctx(ctx) {
    for (auto &in : portInfo.getInputs())
      inputPortMap[in.name] = in.argNum;

    for (auto &out : portInfo.getOutputs())
      outputPortMap[out.name] = out.argNum;
  }

  // Return the index of the input port with the specified name.
  FailureOr<unsigned> getInputPortIndex(StringAttr name) const {
    return lookupPortIndex(inputPortMap, name);
  }

  // Return the index of the output port with the specified name.
  FailureOr<unsigned> getOutputPortIndex(StringAttr name) const {
    return lookupPortIndex(outputPortMap, name);
  }

  FailureOr<unsigned> getInputPortIndex(StringRef name) const {
    return getInputPortIndex(StringAttr::get(ctx, name));
  }

  FailureOr<unsigned> getOutputPortIndex(StringRef name) const {
    return getOutputPortIndex(StringAttr::get(ctx, name));
  }

private:
  llvm::DenseMap<StringAttr, unsigned> inputPortMap;
  llvm::DenseMap<StringAttr, unsigned> outputPortMap;
  MLIRContext *ctx;
};

class InnerSymbolOpInterface;
/// Verification hook for verifying InnerSym Attribute.
LogicalResult verifyInnerSymAttr(InnerSymbolOpInterface op);

namespace detail {
LogicalResult verifyInnerRefNamespace(Operation *op);
} // namespace detail

} // namespace hw
} // namespace circt

namespace mlir {
namespace OpTrait {

/// This trait is for operations that define a scope for resolving InnerRef's,
/// and provides verification for InnerRef users (via InnerRefUserOpInterface).
template <typename ConcreteType>
class InnerRefNamespace : public TraitBase<ConcreteType, InnerRefNamespace> {
public:
  static LogicalResult verifyRegionTrait(Operation *op) {
    static_assert(
        ConcreteType::template hasTrait<::mlir::OpTrait::SymbolTable>(),
        "expected operation to be a SymbolTable");

    if (op->getNumRegions() != 1)
      return op->emitError("expected operation to have a single region");
    if (!op->getRegion(0).hasOneBlock())
      return op->emitError("expected operation to have a single block");

    // Verify all InnerSymbolTable's and InnerRef users.
    return ::circt::hw::detail::verifyInnerRefNamespace(op);
  }
};

/// A trait for inner symbol table functionality on an operation.
template <typename ConcreteType>
class InnerSymbolTable : public TraitBase<ConcreteType, InnerSymbolTable> {
public:
  static LogicalResult verifyRegionTrait(Operation *op) {
    // Insist that ops with InnerSymbolTable's provide a Symbol, this is
    // essential to how InnerRef's work.
    static_assert(
        ConcreteType::template hasTrait<::mlir::SymbolOpInterface::Trait>(),
        "expected operation to define a Symbol");

    // InnerSymbolTable's must be directly nested within an InnerRefNamespace.
    auto *parent = op->getParentOp();
    if (!parent || !parent->hasTrait<InnerRefNamespace>())
      return op->emitError(
          "InnerSymbolTable must have InnerRefNamespace parent");

    return success();
  }
};
} // namespace OpTrait
} // namespace mlir

#include "circt/Dialect/HW/HWOpInterfaces.h.inc"

#endif // CIRCT_DIALECT_HW_HWOPINTERFACES_H
