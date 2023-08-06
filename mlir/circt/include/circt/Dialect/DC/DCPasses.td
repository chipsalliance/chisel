//===- Passes.td - DC dialect passes ----------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_DC_PASSES_TD
#define CIRCT_DIALECT_DC_PASSES_TD

include "mlir/Pass/PassBase.td"

def DCMaterializeForksSinks : InterfacePass<"dc-materialize-forks-sinks", "mlir::FunctionOpInterface"> {
  let summary = "Materialize fork and sink operations.";
  let description = [{
    This pass analyses a function-like operation and inserts fork and sink
    operations ensuring that all values have exactly one use.
  }];
  let constructor = "circt::dc::createDCMaterializeForksSinksPass()";
  let dependentDialects = ["dc::DCDialect"];
}

def DCDematerializeForksSinks : InterfacePass<"dc-dematerialize-forks-sinks", "mlir::FunctionOpInterface"> {
  let summary = "Dematerialize fork and sink operations.";
  let description = [{
    This pass analyses a function-like operation and removes all fork and sink
    operations.
  }];
  let constructor = "circt::dc::createDCDematerializeForksSinksPass()";
  let dependentDialects = ["dc::DCDialect"];
}

#endif // CIRCT_DIALECT_DC_PASSES_TD
