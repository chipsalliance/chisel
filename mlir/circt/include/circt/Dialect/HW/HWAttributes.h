//===- HWAttributes.h - Declare HW dialect attributes ------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_ATTRIBUTES_H
#define CIRCT_DIALECT_HW_ATTRIBUTES_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace circt {
namespace hw {
class PEOAttr;
class EnumType;
enum class PEO : uint32_t;

// Forward declaration.
class GlobalRefOp;

/// Returns a resolved version of 'type' wherein any parameter reference
/// has been evaluated based on the set of provided 'parameters'.
mlir::FailureOr<mlir::Type> evaluateParametricType(mlir::Location loc,
                                                   mlir::ArrayAttr parameters,
                                                   mlir::Type type);

/// Evaluates a parametric attribute (param.decl.ref/param.expr) based on a set
/// of provided parameter values.
mlir::FailureOr<mlir::TypedAttr>
evaluateParametricAttr(mlir::Location loc, mlir::ArrayAttr parameters,
                       mlir::Attribute paramAttr);

/// Returns true if any part of t is parametric.
bool isParametricType(mlir::Type t);

} // namespace hw
} // namespace circt

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/HW/HWAttributes.h.inc"

#endif // CIRCT_DIALECT_HW_ATTRIBUTES_H
