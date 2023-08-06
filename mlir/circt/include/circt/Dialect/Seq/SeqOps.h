//===- SeqOps.h - Declare Seq dialect operations ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the Seq dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SEQ_SEQOPS_H
#define CIRCT_DIALECT_SEQ_SEQOPS_H

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Seq/SeqAttributes.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqOpInterfaces.h"
#include "circt/Dialect/Seq/SeqTypes.h"

#define GET_OP_CLASSES
#include "circt/Dialect/Seq/Seq.h.inc"

namespace circt {
namespace seq {

// Returns true if the given sequence of addresses match the shape of the given
// HLMemType'd handle.
bool isValidIndexValues(Value hlmemHandle, ValueRange addresses);

} // namespace seq
} // namespace circt

#endif // CIRCT_DIALECT_SEQ_SEQOPS_H
