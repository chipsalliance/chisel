//===-- Passes.td - Comb pass definition file --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_COMB_PASSES_TD
#define CIRCT_DIALECT_COMB_PASSES_TD

include "mlir/Pass/PassBase.td"

def LowerComb : Pass<"lower-comb", "mlir::ModuleOp"> {
  let summary = "Lowers the some of the comb ops";
  let description = [{
    Some operations in the comb dialect (e.g. `comb.truth_table`) are not
    directly supported by ExportVerilog. They need to be lowered into ops which
    are supported. There are many ways to lower these ops so we do this in a
    separate pass. This also allows the lowered form to participate in
    optimizations like the comb canonicalizers.
  }];
}

#endif // CIRCT_DIALECT_COMB_PASSES_TD
