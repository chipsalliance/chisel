//===- ArcDialect.td - Arc dialect definition --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ARC_ARCDIALECT_TD
#define CIRCT_DIALECT_ARC_ARCDIALECT_TD

def ArcDialect : Dialect {
  let name = "arc";
  let summary = "Canonical representation of state transfer in a circuit";
  let description = [{
    This is the `arc` dialect, useful for representing state transfer functions
    in a circuit.
  }];
  let cppNamespace = "circt::arc";

  let hasConstantMaterializer = 1;
  let useDefaultTypePrinterParser = 1;

  let extraClassDeclaration = [{
    void registerTypes();
  }];
}

#endif // CIRCT_DIALECT_ARC_ARCDIALECT_TD
