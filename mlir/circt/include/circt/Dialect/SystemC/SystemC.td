//===- SystemC.td - SystemC dialect definition -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the SystemC dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYSTEMC_SYSTEMC
#define CIRCT_DIALECT_SYSTEMC_SYSTEMC

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "circt/Dialect/HW/HWOpInterfaces.td"

include "circt/Dialect/SystemC/SystemCOpInterfaces.td"
include "circt/Dialect/SystemC/SystemCDialect.td"

// Import EmitC types
include "mlir/Dialect/EmitC/IR/EmitCTypes.td"

// Base class for the operations in this dialect.
class SystemCOp<string mnemonic, list<Trait> traits = []> :
  Op<SystemCDialect, mnemonic, traits>;

def HasCustomSSAName : DeclareOpInterfaceMethods<OpAsmOpInterface,
                         ["getAsmResultNames"]>;

include "circt/Dialect/SystemC/SystemCAttributes.td"
include "circt/Dialect/SystemC/SystemCTypes.td"
include "circt/Dialect/SystemC/SystemCExpressions.td"
include "circt/Dialect/SystemC/SystemCStatements.td"
include "circt/Dialect/SystemC/SystemCStructure.td"

#endif // CIRCT_DIALECT_SYSTEMC_SYSTEMC
