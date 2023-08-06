// This test checks that the custom port name gets propagated all the way down to Verilog
// RUN: circt-opt %s \
// RUN:     --lower-scf-to-calyx -canonicalize \
// RUN:     --calyx-remove-comb-groups --canonicalize \
// RUN:     --lower-calyx-to-fsm --canonicalize \
// RUN:     --materialize-calyx-to-fsm --canonicalize \
// RUN:     --calyx-remove-groups-fsm --canonicalize \
// RUN:     --lower-calyx-to-hw --canonicalize \
// RUN:     --convert-fsm-to-sv | FileCheck %s

// CHECK: hw.module @control
// CHECK: hw.module @main(%a: i32, %b: i32, %clk: i1, %reset: i1, %go: i1) -> (out: i1, done: i1)
// CHECK: hw.instance "controller" @control
func.func @main(%arg0 : i32 {calyx.port_name = "a"}, %arg1 : i32 {calyx.port_name = "b"}) -> (i1 {calyx.port_name = "out"}) {
  %0 = arith.cmpi slt, %arg0, %arg1 : i32
  return %0 : i1
}
