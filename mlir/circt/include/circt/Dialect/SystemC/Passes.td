//===-- Passes.td - SystemC pass definition file -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the passes that work on the SystemC dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYSTEMC_PASSES_TD
#define CIRCT_DIALECT_SYSTEMC_PASSES_TD

include "mlir/Pass/PassBase.td"

def SystemCLowerInstanceInterop : Pass<"systemc-lower-instance-interop",
                                       "mlir::ModuleOp"> {
  let summary = "Lower all SystemC instance interop operations.";

  let constructor =  "circt::systemc::createSystemCLowerInstanceInteropPass()";
  let dependentDialects = [
    "systemc::SystemCDialect", "interop::InteropDialect",
    "mlir::emitc::EmitCDialect", "mlir::func::FuncDialect"
  ];
}

#endif // CIRCT_DIALECT_SYSTEMC_PASSES_TD
