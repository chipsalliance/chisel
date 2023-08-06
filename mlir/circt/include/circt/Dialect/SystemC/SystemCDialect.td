//===- SystemCDialect.td - SystemC dialect definition ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains the SystemC dialect definition to be included in other files.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYSTEMC_SYSTEMCDIALECT
#define CIRCT_DIALECT_SYSTEMC_SYSTEMCDIALECT

def SystemCDialect : Dialect {
  let name = "systemc";
  let cppNamespace = "::circt::systemc";

  let summary = "Types and operations for the SystemC dialect";
  let description = [{
    This dialect defines the `systemc` dialect, which represents various
    constructs of the SystemC library (IEEE 1666-2011) useful for emission.
  }];

  let extraClassDeclaration = [{
    /// Register all SystemC attributes.
    void registerAttributes();
    /// Register all SystemC types.
    void registerTypes();

    /// Type parsing and printing.
    mlir::Type parseType(mlir::DialectAsmParser &parser) const override;
    void printType(mlir::Type, mlir::DialectAsmPrinter &) const override;
  }];

  let useDefaultTypePrinterParser = 0;
}

#endif // CIRCT_DIALECT_SYSTEMC_SYSTEMCDIALECT
