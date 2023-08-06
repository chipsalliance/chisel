//===- LLHDTypes.h - Types for the LLHD dialect -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Types for the LLHD dialect are mostly in tablegen. This file should contain
// C++ types used in MLIR type parameters.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_LLHD_IR_LLHDTYPES_H
#define CIRCT_DIALECT_LLHD_IR_LLHDTYPES_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/LLHD/IR/LLHDTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/LLHD/IR/LLHDAttributes.h.inc"

#endif // CIRCT_DIALECT_LLHD_IR_LLHDTYPES_H
