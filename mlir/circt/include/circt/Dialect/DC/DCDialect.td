//===- DCDialect.td - DC dialect definition ----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_DC_DIALECT_TD
#define CIRCT_DIALECT_DC_DIALECT_TD

include "mlir/IR/OpBase.td"

def DCDialect : Dialect {
  let name = "dc";
  let summary = "Dynamic Control";
  let description = [{
    This is the `dc` dialect, used to represent dynamic control constructs with
    handshaking semantics.
  }];
  let cppNamespace = "circt::dc";

  let useDefaultTypePrinterParser = 1;
  let dependentDialects = ["circt::esi::ESIDialect"];

  let extraClassDeclaration = [{
    void registerTypes();
  }];
}

#endif // CIRCT_DIALECT_DC_DIALECT_TD
