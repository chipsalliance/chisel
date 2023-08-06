//===- VerifDialect.td - Verif dialect definition ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_VERIF_VERIFDIALECT_TD
#define CIRCT_DIALECT_VERIF_VERIFDIALECT_TD

include "mlir/IR/OpBase.td"

def VerifDialect : Dialect {
  let name = "verif";
  let summary = "Verification constructs and utilities.";
  // See `docs/Dialect/Verif.md` for detailed dialect documentation.
  let cppNamespace = "circt::verif";
  let hasConstantMaterializer = 1;
}

#endif // CIRCT_DIALECT_VERIF_VERIFDIALECT_TD
