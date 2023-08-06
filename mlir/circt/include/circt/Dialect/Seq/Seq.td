//===- Seq.td - Seq dialect definition ---------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the Seq dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SEQ_SEQ_TD
#define CIRCT_DIALECT_SEQ_SEQ_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "circt/Dialect/Seq/SeqDialect.td"
include "circt/Dialect/Seq/SeqAttributes.td"
include "circt/Dialect/Seq/SeqTypes.td"
include "circt/Dialect/Seq/SeqOps.td"

#endif // CIRCT_DIALECT_SEQ_SEQ_TD
