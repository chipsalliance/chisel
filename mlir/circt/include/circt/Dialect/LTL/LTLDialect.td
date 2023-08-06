//===- LTLDialect.td - LTL dialect definition --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_LTL_LTLDIALECT_TD
#define CIRCT_DIALECT_LTL_LTLDIALECT_TD

include "mlir/IR/OpBase.td"

def LTLDialect : Dialect {
  let name = "ltl";
  let summary = "Linear temporal logic, sequences, and properties.";
  // See `docs/Dialect/LTL.md` for detailed dialect documentation.
  let cppNamespace = "circt::ltl";
  let useDefaultTypePrinterParser = 1;
  let dependentDialects = ["hw::HWDialect", "comb::CombDialect"];
}

#endif // CIRCT_DIALECT_LTL_LTLDIALECT_TD
