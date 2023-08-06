//===- ESITypes.h - types for the ESI dialect -------------------*- C++ -*-===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Types for ESI are mostly in tablegen. This file should contain C++ types used
// in MLIR type parameters and other supporting declarations.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ESI_ESITYPES_H
#define CIRCT_DIALECT_ESI_ESITYPES_H

#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"

#include "ESIDialect.h"

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/ESI/ESITypes.h.inc"

namespace circt {
namespace esi {

// If 'type' is an esi:ChannelType, will return the inner type of said channel.
// Else, returns 'type'.
mlir::Type innerType(mlir::Type type);
} // namespace esi
} // namespace circt

#endif
