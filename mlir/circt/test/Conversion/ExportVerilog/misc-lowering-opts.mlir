// RUN: circt-opt %s -export-verilog --split-input-file | FileCheck %s

!fooTy = !hw.struct<bar: i4>

module attributes {circt.loweringOptions="emitWireInPorts"} {
// CHECK-LABEL: module Foo(
// CHECK-NEXT:    input  wire                                   a,
// CHECK-NEXT:    input  wire struct packed {logic [3:0] bar; } foo,
// CHECK-NEXT:    output wire [2:0]                             x	
// CHECK:       endmodule
hw.module @Foo(%a: i1, %foo: !fooTy) -> (x: i3) {
  %c0_i3 = hw.constant 0 : i3
  hw.output %c0_i3 : i3
} 
}
