//===- HWDialect.td - HW dialect definition ----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains the HWDialect definition to be included in other files.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_HWDIALECT
#define CIRCT_DIALECT_HW_HWDIALECT

include "mlir/IR/OpBase.td"

def HWDialect : Dialect {
  let name = "hw";
  let cppNamespace = "::circt::hw";

  let summary = "Types and operations for the hardware dialect";
  let description = [{
    This dialect defines the `hw` dialect, which is intended to be a generic
    representation of HW outside of a particular use-case.
  }];

  let hasConstantMaterializer = 1;
  let useDefaultTypePrinterParser = 1;

  let extraClassDeclaration = [{
    /// Register all HW types.
    void registerTypes();
    /// Register all HW attributes.
    void registerAttributes();

    Attribute parseAttribute(DialectAsmParser &p, Type type) const override;
    void printAttribute(Attribute attr, DialectAsmPrinter &p) const override;
  }];
}

// Base class for the operation in this dialect.
class HWOp<string mnemonic, list<Trait> traits = []> :
    Op<HWDialect, mnemonic, traits>;

#endif // CIRCT_DIALECT_HW_HWDIALECT
