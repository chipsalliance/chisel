//===-- Passes.td - FSM pass definition file ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the passes that work on the FSM dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FSM_PASSES_TD
#define CIRCT_DIALECT_FSM_PASSES_TD

include "mlir/Pass/PassBase.td"

def PrintFSMGraph : Pass<"fsm-print-graph", "mlir::ModuleOp"> {
  let summary = "Print a DOT graph of the module hierarchy.";
  let constructor =  "circt::fsm::createPrintFSMGraphPass()";
}

#endif // CIRCT_DIALECT_FSM_PASSES_TD
