// REQUIRES: verilator
// RUN: circt-opt %s -export-verilog -verify-diagnostics -o %t2.mlir > %t1.sv
// RUN: circt-rtl-sim.py %t1.sv --cycles 8 2>&1 | FileCheck %s

module {
  // The HW dialect doesn't have any sequential constructs yet. So don't do
  // much.
  hw.module @top(%clk: i1, %rst: i1) {
    %c1 = hw.instance "aaa" @AAA () -> (f: i1)
    %c1Shl = hw.instance "shl" @shl (a: %c1: i1) -> (b: i1)
    sv.always posedge %clk {
      %fd = hw.constant 0x80000002 : i32
      sv.fwrite %fd, "tick\n"
    }
  }

  hw.module @AAA() -> (f: i1) {
    %z = hw.constant 1 : i1
    hw.output %z : i1
  }

  hw.module @shl(%a: i1) -> (b: i1) {
    %0 = comb.shl %a, %a : i1
    hw.output %0 : i1
  }
}

// CHECK:      [driver] Starting simulation
// CHECK-NEXT: tick
// CHECK-NEXT: tick
// CHECK-NEXT: tick
// CHECK-NEXT: tick
// CHECK-NEXT: tick
// CHECK-NEXT: tick
// CHECK-NEXT: tick
// CHECK-NEXT: tick
// CHECK-NEXT: tick
// CHECK-NEXT: tick
// CHECK-NEXT: tick
// CHECK-NEXT: tick
// CHECK-NEXT: [driver] Ending simulation at tick #25
