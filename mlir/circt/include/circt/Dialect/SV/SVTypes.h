//===- SVTypes.h - Declare SV dialect types ----------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the types for the SV dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SV_TYPES_H
#define CIRCT_DIALECT_SV_TYPES_H

#include "circt/Dialect/HW/HWTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/SV/SVTypes.h.inc"

namespace circt {
namespace sv {
using InOutType = circt::hw::InOutType;

/// Return the element type of an InOutType or null if the operand isn't an
/// InOut type.
mlir::Type getInOutElementType(mlir::Type type);

/// Return the element type of an ArrayType or UnpackedArrayType, or null if the
/// operand isn't an array.
mlir::Type getAnyHWArrayElementType(mlir::Type type);
} // end namespace sv
} // end namespace circt

#endif // CIRCT_DIALECT_SV_TYPES_H
