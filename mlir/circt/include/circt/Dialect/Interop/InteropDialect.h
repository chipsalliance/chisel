//===- InteropDialect.h - Interop dialect declaration -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an Interop MLIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_INTEROP_INTEROPDIALECT_H
#define CIRCT_DIALECT_INTEROP_INTEROPDIALECT_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/Dialect.h"

// Pull in the dialect definition.
#include "circt/Dialect/Interop/InteropDialect.h.inc"

#endif // CIRCT_DIALECT_INTEROP_INTEROPDIALECT_H
