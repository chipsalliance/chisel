//===- OMTypes.h - Object Model type declarations -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the Object Model type declarations.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_OM_OMTYPES_H
#define CIRCT_DIALECT_OM_OMTYPES_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/OM/OMTypes.h.inc"

#endif // CIRCT_DIALECT_OM_OMTYPES_H
