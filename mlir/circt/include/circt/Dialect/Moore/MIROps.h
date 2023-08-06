//===- MIROps.h - Declare Moore MIR dialect operations ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the Moore MIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_MOORE_MIROPS_H
#define CIRCT_DIALECT_MOORE_MIROPS_H

#include "circt/Dialect/Moore/MooreDialect.h"
#include "circt/Dialect/Moore/MooreTypes.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

#define GET_OP_CLASSES
#include "circt/Dialect/Moore/MooreEnums.h.inc"
// Clang format shouldn't reorder these headers.
#include "circt/Dialect/Moore/Moore.h.inc"

#endif // CIRCT_DIALECT_MOORE_MIROPS_H
