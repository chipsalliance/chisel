//===- MSFTPasses.td - MSFT dialect passes -----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "mlir/Pass/PassBase.td"

def LowerToHW: Pass<"lower-msft-to-hw", "mlir::ModuleOp"> {
  let summary = "Lower MSFT ops to hw ops";
  let constructor = "circt::msft::createLowerToHWPass()";
  let dependentDialects = ["circt::sv::SVDialect", "circt::hw::HWDialect"];
  let options = [
    Option<"verilogFile", "verilog-file", "std::string",
           "", "File to output Verilog into">,
  ];
}

def ExportTcl: Pass<"msft-export-tcl", "mlir::ModuleOp"> {
  let summary = "Create tcl ops";
  let constructor = "circt::msft::createExportTclPass()";
  let dependentDialects = ["circt::sv::SVDialect", "circt::hw::HWDialect"];
  let options = [
    ListOption<"tops", "tops", "std::string",
               "List of top modules to export Tcl for",
               "llvm::cl::ZeroOrMore,">,
    Option<"tclFile", "tcl-file", "std::string",
           "", "File to output Tcl into">
  ];
}

def Partition: Pass<"msft-partition", "mlir::ModuleOp"> {
  let summary = "Move the entities targeted for a design partition";
  let constructor = "circt::msft::createPartitionPass()";
  let dependentDialects = ["circt::hw::HWDialect"];
}

def WireCleanup: Pass<"msft-wire-cleanup", "mlir::ModuleOp"> {
  let summary = "Cleanup unnecessary ports and wires";
  let constructor = "circt::msft::createWireCleanupPass()";
  let dependentDialects = [];
}

def LowerInstances: Pass<"msft-lower-instances", "mlir::ModuleOp"> {
  let summary = "Lower dynamic instances";
  let constructor = "circt::msft::createLowerInstancesPass()";
  let dependentDialects = ["circt::hw::HWDialect"];
}

def LowerConstructs: Pass<"msft-lower-constructs", "mlir::ModuleOp"> {
  let summary = "Lower high-level constructs";
  let constructor = "circt::msft::createLowerConstructsPass()";
  let dependentDialects = ["circt::hw::HWDialect"];
}

def DiscoverAppIDs: Pass<"msft-discover-appids", "mlir::ModuleOp"> {
  let summary = "Discover the appids in a module hierarchy";
  let constructor = "circt::msft::createDiscoverAppIDsPass()";
}
