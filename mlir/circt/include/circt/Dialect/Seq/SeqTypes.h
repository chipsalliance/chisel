//===- SeqTypes.h - Types for the Seq dialect -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Types for the Seq dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SEQ_SEQTYPES_H
#define CIRCT_DIALECT_SEQ_SEQTYPES_H

#include "circt/Dialect/Seq/SeqDialect.h"

#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/Seq/SeqTypes.h.inc"

#endif // CIRCT_DIALECT_SEQ_SEQTYPES_H
