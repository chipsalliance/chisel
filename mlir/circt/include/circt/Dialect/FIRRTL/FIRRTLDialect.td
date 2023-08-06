//===- FIRRTLDialect.td - FIRRTL dialect definition --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the FIRRTL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_FIRRTLDIALECT_TD
#define CIRCT_DIALECT_FIRRTL_FIRRTLDIALECT_TD

include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"

def HasCustomSSAName : DeclareOpInterfaceMethods<OpAsmOpInterface,
                         ["getAsmResultNames"]>;

def FIRRTLDialect : Dialect {
  let name = "firrtl";
  let cppNamespace = "::circt::firrtl";

  let summary = "Types and operations for firrtl dialect";
  let description = [{
    This dialect defines the `firrtl` dialect, which is used to lower from Chisel
    code to Verilog.  For more information, see the [FIRRTL GitHub
    page](https://github.com/freechipsproject/firrtl).
  }];

  let hasConstantMaterializer = 1;
  let useDefaultTypePrinterParser = 0;
  let useDefaultAttributePrinterParser = 1;

  let dependentDialects = [
    "circt::hw::HWDialect",
    "circt::om::OMDialect"
  ];

  let extraClassDeclaration = [{
     ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;
     void printType(::mlir::Type type, ::mlir::DialectAsmPrinter &printer) const override;

    /// Register all FIRRTL types.
    void registerTypes();
    /// Register all attributes.
    void registerAttributes();
  }];
}

// Base class for the operation in this dialect.
class FIRRTLOp<string mnemonic, list<Trait> traits = []> :
    Op<FIRRTLDialect, mnemonic, traits>;

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLDIALECT_TD
