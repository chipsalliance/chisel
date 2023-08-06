// RUN: circt-opt --sv-trace-iverilog --export-verilog %s | FileCheck %s

// CHECK-LABEL: module top();
// CHECK:       initial begin
// CHECK-NEXT:    $dumpfile ("./top.vcd");
// CHECK-NEXT:    $dumpvars (0, top);
// CHECK-NEXT:    #1;
// CHECK-NEXT:  end

hw.module @top () {}
