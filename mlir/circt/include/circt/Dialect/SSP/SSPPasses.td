//===- SSPPasses.td - SSP pass definition file -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the passes that work on the SSP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SSP_SSPPASSES_TD
#define CIRCT_DIALECT_SSP_SSPPASSES_TD

include "mlir/Pass/PassBase.td"

def Print : Pass<"ssp-print", "mlir::ModuleOp"> {
  let summary = "Prints all SSP instances as DOT graphs.";
  let constructor = "circt::ssp::createPrintPass()";
}

def Roundtrip : Pass<"ssp-roundtrip", "mlir::ModuleOp"> {
  let summary = "Roundtrips all SSP instances via the scheduling infrastructure";
  let constructor = "circt::ssp::createRoundtripPass()";
  let options = [
    Option<"checkInputConstraints", "check", "bool", "false",
           "Check the problem's input constraints.">,
    Option<"verifySolutionConstraints", "verify", "bool", "false",
           "Verify the problem's solution constraints.">,
  ];
}

def Schedule : Pass<"ssp-schedule", "mlir::ModuleOp"> {
  let summary = "Schedules all SSP instances.";
  let constructor = "circt::ssp::createSchedulePass()";
  let options = [
    Option<"scheduler", "scheduler", "std::string", "",
           "Scheduling algorithm to use.">,
    Option<"schedulerOptions", "options", "std::string", "",
           "Scheduler-specific options.">
  ];
}

#endif // CIRCT_DIALECT_SSP_SSPPASSES_TD
