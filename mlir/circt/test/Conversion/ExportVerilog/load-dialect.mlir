// RUN: circt-opt %s -export-verilog | FileCheck %s
// https://github.com/llvm/circt/issues/854
// This is testing that dependent dialects are explicitly loaded. ExportVerilog
// must explicitly load any dialect that it creates operations for. For this
// test to work as intended:
//   1. the IR must not have any operations from the SV dialect
//   2. the IR must trigger ExportVerilog to create a sv.wire


hw.module @cyclic(%a: i1) -> (b: i1) {
  // Check that a wire temporary is created by export verilog. This wire is
  // for holding the value of %0.  If this wire is not emitted then this test
  // should either be deleted or find a different way to force IR generation.

  // CHECK: wire _GEN;

  %1 = comb.add %0, %0 : i1
  %0 = comb.shl %a, %a : i1
  %2 = comb.add %1, %1 : i1
  hw.output %2 : i1
}
