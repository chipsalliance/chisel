//===- CalyxOps.h - Declare Calyx dialect operations ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation class for the Calyx IR.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_CALYX_OPS_H
#define CIRCT_DIALECT_CALYX_OPS_H

#include "circt/Dialect/Calyx/CalyxDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace circt {
namespace calyx {

/// A helper function to verify each control-like operation
/// has a valid parent and, if applicable, body.
LogicalResult verifyControlLikeOp(Operation *op);

/// Signals that the following operation is "control-like."
template <typename ConcreteType>
class ControlLike : public mlir::OpTrait::TraitBase<ConcreteType, ControlLike> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return verifyControlLikeOp(op);
  }
};

/// A helper function to verify a combinational operation.
LogicalResult verifyCombinationalOp(Operation *op);

/// Signals that the following operation is combinational.
template <typename ConcreteType>
class Combinational
    : public mlir::OpTrait::TraitBase<ConcreteType, Combinational> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    Attribute staticAttribute = op->getAttr("static");
    if (staticAttribute == nullptr)
      return success();

    // If the operation has the static attribute, verify it is zero.
    APInt staticValue = staticAttribute.cast<IntegerAttr>().getValue();
    assert(staticValue == 0 && "If combinational, it should take 0 cycles.");

    return success();
  }
};

/// The direction of a Component or Cell port. this is similar to the
/// implementation found in the FIRRTL dialect.
enum Direction { Input = 0, Output = 1 };
namespace direction {
/// Returns an output direction if `isOutput` is true, otherwise returns an
/// input direction.
Direction get(bool isOutput);

/// Returns an IntegerAttr containing the packed representation of the
/// direction counts. Direction::Input is zero, and Direction::Output is one.
IntegerAttr packAttribute(MLIRContext *context, size_t nIns, size_t nOuts);

} // namespace direction

/// This holds information about the port for either a Component or Cell.
struct PortInfo {
  StringAttr name;
  Type type;
  Direction direction;
  DictionaryAttr attributes;

  /// Returns whether the given port has attribute with Identifier `name`.
  bool hasAttribute(StringRef identifier) const {
    assert(attributes && "PortInfo::attributes should be instantiated.");
    return llvm::any_of(attributes, [&](auto idToAttribute) {
      return identifier == idToAttribute.getName();
    });
  }

  /// Returns the attribute associated with the given name if it exists,
  /// otherwise std::nullopt.
  std::optional<Attribute> getAttribute(StringRef identifier) const {
    assert(attributes && "PortInfo::attributes should be instantiated.");
    auto it = llvm::find_if(attributes, [&](auto idToAttribute) {
      return identifier == idToAttribute.getName();
    });
    if (it == attributes.end())
      return std::nullopt;
    return it->getValue();
  }

  /// Returns all identifiers for this dictionary attribute.
  SmallVector<StringRef> getAllIdentifiers() const {
    assert(attributes && "PortInfo::attributes should be instantiated.");
    SmallVector<StringRef> identifiers;
    llvm::transform(attributes, std::back_inserter(identifiers),
                    [](auto idToAttribute) { return idToAttribute.getName(); });
    return identifiers;
  }
};

/// A helper function to verify each operation with the Ccomponent trait.
LogicalResult verifyComponent(Operation *op);

/// A helper function to verify each operation with the Cell trait.
LogicalResult verifyCell(Operation *op);

/// A helper function to verify each operation with the Group Interface trait.
LogicalResult verifyGroupInterface(Operation *op);

/// A helper function to verify each operation with the If trait.
LogicalResult verifyIf(Operation *op);

/// Returns port information for the block argument provided.
PortInfo getPortInfo(BlockArgument arg);

} // namespace calyx
} // namespace circt

#include "circt/Dialect/Calyx/CalyxInterfaces.h.inc"

#define GET_OP_CLASSES
#include "circt/Dialect/Calyx/Calyx.h.inc"

#endif // CIRCT_DIALECT_CALYX_OPS_H
