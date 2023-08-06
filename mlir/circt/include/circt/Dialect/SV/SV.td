//===- SV.td - SystemVerilog dialect definition ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the SV dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SV_SV
#define CIRCT_DIALECT_SV_SV

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "circt/Dialect/SV/SVDialect.td"

// Base class for the operation in this dialect.
class SVOp<string mnemonic, list<Trait> traits = []> :
    Op<SVDialect, mnemonic, traits>;

def ProceduralRegion : NativeOpTrait<"ProceduralRegion"> {
  let cppNamespace = "::circt::sv";
}

def ProceduralOp : NativeOpTrait<"ProceduralOp"> {
  let cppNamespace = "::circt::sv";
}

def NonProceduralOp : NativeOpTrait<"NonProceduralOp"> {
  let cppNamespace = "::circt::sv";
}

/// Mark an operation as being a vendor extension.
def VendorExtension : NativeOpTrait<"VendorExtension"> {
  let cppNamespace = "::circt::sv";
}

include "circt/Dialect/HW/HWTypes.td"
include "circt/Dialect/HW/HWAttributesNaming.td"
include "circt/Dialect/SV/SVAttributes.td"
include "circt/Dialect/SV/SVTypes.td"
include "circt/Dialect/SV/SVExpressions.td"
include "circt/Dialect/SV/SVInOutOps.td"
include "circt/Dialect/SV/SVGenerate.td"
include "circt/Dialect/SV/SVStatements.td"
include "circt/Dialect/SV/SVVerification.td"
include "circt/Dialect/SV/SVTypeDecl.td"

#endif // CIRCT_DIALECT_SV_SV
