//===- SystemCAttributes.h - Declare SystemC dialect attributes --*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the attributes for the SystemC dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYSTEMC_SYSTEMCATTRIBUTES_H
#define CIRCT_DIALECT_SYSTEMC_SYSTEMCATTRIBUTES_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/APInt.h"

// Include generated attributes.
#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/SystemC/SystemCEnums.h.inc"
// do not reorder
#include "circt/Dialect/SystemC/SystemCAttributes.h.inc"

#endif // CIRCT_DIALECT_SYSTEMC_SYSTEMCATTRIBUTES_H
