//===- FIRRTLOpInterfaces.h - Declare FIRRTL op interfaces ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation interfaces for the FIRRTL IR and supporting
// types.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_OP_INTERFACES_H
#define CIRCT_DIALECT_FIRRTL_OP_INTERFACES_H

#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/InnerSymbolTable.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
class PatternRewriter;
} // end namespace mlir

namespace circt {
namespace firrtl {

class FIRRTLType;
class Forceable;

/// This holds the name and type that describes the module's ports.
struct PortInfo {
  StringAttr name;
  Type type;
  Direction direction;
  hw::InnerSymAttr sym = {};
  Location loc = UnknownLoc::get(type.getContext());
  AnnotationSet annotations = AnnotationSet(type.getContext());

  StringRef getName() const { return name ? name.getValue() : ""; }

  /// Return true if this is a simple output-only port.  If you want the
  /// direction of the port, use the \p direction parameter.
  bool isOutput() const { return direction == Direction::Out && !isInOut(); }

  /// Return true if this is a simple input-only port.  If you want the
  /// direction of the port, use the \p direction parameter.
  bool isInput() const { return direction == Direction::In && !isInOut(); }

  /// Return true if this is an inout port.  This will be true if the port
  /// contains either bi-directional signals or analog types.
  /// Non-HW types (e.g., ref types) are never considered InOut.
  bool isInOut() const { return isTypeInOut(type); }

  /// Default constructors
  PortInfo(StringAttr name, Type type, Direction dir, StringAttr symName = {},
           std::optional<Location> location = {},
           std::optional<AnnotationSet> annos = {})
      : name(name), type(type), direction(dir) {
    if (symName)
      sym = hw::InnerSymAttr::get(symName);
    if (location)
      loc = *location;
    if (annos)
      annotations = *annos;
  };
  PortInfo(StringAttr name, Type type, Direction dir, hw::InnerSymAttr sym,
           Location loc, AnnotationSet annos)
      : name(name), type(type), direction(dir), sym(sym), loc(loc),
        annotations(annos) {}
};

enum class ConnectBehaviorKind {
  /// Classic FIRRTL connections: last connect 'wins' across paths;
  /// conditionally applied under 'when'.
  LastConnect,
  /// Exclusive connection to the destination, unconditional.
  StaticSingleConnect,
};

/// Verification hook for verifying module like operations.
LogicalResult verifyModuleLikeOpInterface(FModuleLike module);

namespace detail {
/// Return null or forceable reference result type.
RefType getForceableResultType(bool forceable, Type type);
/// Verify a Forceable op.
LogicalResult verifyForceableOp(Forceable op);
/// Replace a Forceable op with equivalent, changing whether forceable.
/// No-op if already has specified forceability.
Forceable
replaceWithNewForceability(Forceable op, bool forceable,
                           ::mlir::PatternRewriter *rewriter = nullptr);
} // end namespace detail

} // namespace firrtl
} // namespace circt

#include "circt/Dialect/FIRRTL/FIRRTLOpInterfaces.h.inc"
#endif // CIRCT_DIALECT_FIRRTL_OP_INTERFACES_H
