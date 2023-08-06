//===- FoldUtils.h - Common folder and canonicalizer utilities --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_FOLDUTILS_H
#define CIRCT_SUPPORT_FOLDUTILS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/APInt.h"

namespace circt {

/// Determine the integer value of a constant operand.
static inline std::optional<APInt> getConstantInt(Attribute operand) {
  if (!operand)
    return {};
  if (auto attr = dyn_cast<IntegerAttr>(operand))
    return attr.getValue();
  return {};
}

/// Determine whether a constant operand is a zero value.
static inline bool isConstantZero(Attribute operand) {
  if (auto cst = getConstantInt(operand))
    return cst->isZero();
  return false;
}

/// Determine whether a constant operand is a one value.
static inline bool isConstantOne(Attribute operand) {
  if (auto cst = getConstantInt(operand))
    return cst->isOne();
  return false;
}

} // namespace circt

#endif // CIRCT_SUPPORT_FOLDUTILS_H
