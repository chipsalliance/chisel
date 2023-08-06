//===- MooreDialect.td - Moore dialect definition ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains the Moore dialect definition to be included in other files.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_MOORE_MOOREDIALECT
#define CIRCT_DIALECT_MOORE_MOOREDIALECT

def MooreDialect : Dialect {
  let name = "moore";

  let summary = "Types and operations for Moore dialect";
  let description = [{
    This dialect defines the `moore` dialect, which represents various
    SystemVerilog-specific constructs without ambiguities and all types
    resolved.
  }];
  let cppNamespace = "::circt::moore";
  let extraClassDeclaration = [{
    /// Register all Moore types.
    void registerTypes();

    /// Type parsing and printing.
    Type parseType(DialectAsmParser &parser) const override;
    void printType(Type, DialectAsmPrinter &) const override;
  }];
  let useDefaultTypePrinterParser = 0;
}

#endif // CIRCT_DIALECT_MOORE_MOOREDIALECT
