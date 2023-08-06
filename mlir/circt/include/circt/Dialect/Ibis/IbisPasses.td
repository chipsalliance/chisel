//===-- Passes.td - Ibis pass definition file --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_IBIS_PASSES_TD
#define CIRCT_DIALECT_IBIS_PASSES_TD

include "mlir/Pass/PassBase.td"

def IbisCallPrep : Pass<"ibis-call-prep", "ModuleOp"> {
  let summary = "Convert ibis method calls to use `dc.value`";

  let constructor = "circt::ibis::createCallPrepPass()";
  let dependentDialects = [
    "circt::hw::HWDialect", "circt::dc::DCDialect"];
}

#endif // CIRCT_DIALECT_IBIS_PASSES_TD
