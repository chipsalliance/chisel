//===-- SeqPasses.td - Seq pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for passes that work on the Seq dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SEQ_SEQPASSES
#define CIRCT_DIALECT_SEQ_SEQPASSES

include "mlir/Pass/PassBase.td"

def LowerSeqToSV: Pass<"lower-seq-to-sv", "mlir::ModuleOp"> {
  let summary = "Lower sequential ops to SV.";
  let constructor = "circt::seq::createSeqLowerToSVPass()";
  let dependentDialects = ["circt::sv::SVDialect"];
  let options = [
    Option<"lowerToAlwaysFF", "lower-to-always-ff", "bool", "true",
           "Place assignments to registers into `always_ff` blocks">
  ];
}

def LowerSeqFIRRTLInitToSV: Pass<"lower-seq-firrtl-init-to-sv", "mlir::ModuleOp"> {
  let summary = "Prep the module with macro definitions for firrtl registers.";
  let constructor = "circt::seq::createLowerSeqFIRRTLInitToSV()";
  let dependentDialects = ["circt::sv::SVDialect"];
}

def LowerSeqFIRRTLToSV: Pass<"lower-seq-firrtl-to-sv",  "hw::HWModuleOp"> {
  let summary = "Lower sequential firrtl ops to SV.";
  let constructor = "circt::seq::createSeqFIRRTLLowerToSVPass()";
  let dependentDialects = ["circt::sv::SVDialect"];
  let options = [
    Option<"disableRegRandomization", "disable-reg-randomization", "bool", "false",
           "Disable emission of register randomization code">,
    Option<"emitSeparateAlwaysBlocks", "emit-separate-always-blocks", "bool", "false",
           "Emit assigments to registers in separate always blocks">
  ];
  let statistics = [
    Statistic<"numSubaccessRestored", "num-subaccess-restored",
      "Number of lhs subaccess operations restored ">
  ];
}

def LowerSeqFIFO : Pass<"lower-seq-fifo", "hw::HWModuleOp"> {
  let summary = "Lower seq.fifo ops";
  let constructor = "circt::seq::createLowerSeqFIFOPass()";
  let dependentDialects = ["circt::hw::HWDialect", "circt::comb::CombDialect"];
}

def LowerSeqHLMem: Pass<"lower-seq-hlmem", "hw::HWModuleOp"> {
  let summary = "Lowers seq.hlmem operations.";
  let constructor = "circt::seq::createLowerSeqHLMemPass()";
  let dependentDialects = ["circt::sv::SVDialect"];
}

def ExternalizeClockGate: Pass<"externalize-clock-gate", "mlir::ModuleOp"> {
  let summary = "Convert seq.clock_gate ops into hw.module.extern instances";
  let constructor = "circt::seq::createExternalizeClockGatePass()";
  let dependentDialects = ["circt::hw::HWDialect", "circt::comb::CombDialect"];
  let options = [
    Option<"moduleName", "name", "std::string", "\"CKG\"",
           "Name of the external clock gate module">,
    Option<"inputName", "input", "std::string", "\"I\"",
           "Name of the clock input">,
    Option<"outputName", "output", "std::string", "\"O\"",
           "Name of the gated clock output">,
    Option<"enableName", "enable", "std::string", "\"E\"",
           "Name of the enable input">,
    Option<"testEnableName", "test-enable", "std::string", "\"TE\"",
           "Name of the optional test enable input">,
    Option<"instName", "instance-name", "std::string", "\"ckg\"",
           "Name of the generated instances">
  ];
  let statistics = [
    Statistic<"numClockGatesConverted", "num-clock-gates-converted",
      "Number of clock gates converted to external module instances">
  ];
}

def LowerFirMem : Pass<"lower-seq-firmem", "mlir::ModuleOp"> {
  let summary = "Lower seq.firmem ops to instances of hw.module.generated ops";
  let constructor = "circt::seq::createLowerFirMemPass()";
  let dependentDialects = ["circt::hw::HWDialect"];
}

#endif // CIRCT_DIALECT_SEQ_SEQPASSES
