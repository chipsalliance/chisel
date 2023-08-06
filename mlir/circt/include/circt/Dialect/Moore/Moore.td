//===- Moore.td - Moore dialect definition -----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the Moore dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_MOORE_MOORE
#define CIRCT_DIALECT_MOORE_MOORE

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "circt/Dialect/Moore/MooreDialect.td"

// Base class for the operations in this dialect.
class MooreOp<string mnemonic, list<Trait> traits = []> :
  Op<MooreDialect, mnemonic, traits>;

// Base class for the MIR operations in this dialect.
class MIROp<string mnemonic, list<Trait> traits = []> :
  MooreOp<"mir." # mnemonic, traits>;

include "circt/Dialect/Moore/MooreTypesImpl.td"
include "circt/Dialect/Moore/MooreTypes.td"
include "circt/Dialect/Moore/MIRExpressions.td"
include "circt/Dialect/Moore/MIRStatements.td"

#endif // CIRCT_DIALECT_MOORE_MOORE
