//===- Comb.td - Comb dialect definition --------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the Comb dialect.
//
//===----------------------------------------------------------------------===//

#ifndef COMB_TD
#define COMB_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"

def CombDialect : Dialect {
  let name = "comb";

  let summary = "Types and operations for comb dialect";
  let description = [{
    This dialect defines the `comb` dialect, which is intended to be a generic
    representation of combinational logic outside of a particular use-case.
  }];
  let hasConstantMaterializer = 1;
  let cppNamespace = "::circt::comb";
}

// Base class for the operation in this dialect.
class CombOp<string mnemonic, list<Trait> traits = []> :
    Op<CombDialect, mnemonic, traits>;

include "circt/Dialect/HW/HWTypes.td"
include "circt/Dialect/Comb/Combinational.td"

#endif // COMB_TD
