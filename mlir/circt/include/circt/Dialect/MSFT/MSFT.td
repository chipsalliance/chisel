//===- MSFT.td - MSFT dialect definition -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the MSFT dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MSFT_TD
#define MSFT_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Pass/PassBase.td"

def MSFTDialect : Dialect {
  let name = "msft";
  let cppNamespace = "::circt::msft";

  let summary = "Microsoft internal support dialect";
  let description = [{
    Umbrella dialect for everything needed to support Microsoft development
    but not thoroughly discussed. Most (if not everything) in this dialect is
    a candidate for generalization and re-homing.
  }];

  let hasConstantMaterializer = 1;
  let useDefaultAttributePrinterParser = 1;

  let extraClassDeclaration = [{
    /// Register all MSFT attributes.
    void registerAttributes();
  }];

  let dependentDialects = ["::circt::hw::HWDialect"];
}
// Base class for the operation in this dialect.
class MSFTOp<string mnemonic, list<Trait> traits = []> :
    Op<MSFTDialect, mnemonic, traits>;

include "circt/Dialect/MSFT/MSFTAttributes.td"
include "circt/Dialect/MSFT/MSFTOpInterfaces.td"
include "circt/Dialect/MSFT/MSFTOps.td"
include "circt/Dialect/MSFT/MSFTPDOps.td"
include "circt/Dialect/MSFT/MSFTConstructs.td"
include "circt/Dialect/MSFT/MSFTPasses.td"

#endif // MSFT_TD
