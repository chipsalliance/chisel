// RUN: circt-translate --export-calyx --split-input-file --verify-diagnostics %s | FileCheck %s --strict-whitespace

module attributes {calyx.entrypoint = "main"} {
  // CHECK: import "primitives/memories.futil";
  // CHECK-LABEL: component main<"static"=1>(in: 32, @go go: 1, @clk clk: 1, @reset reset: 1) -> (out: 32, @done done: 1) {
  calyx.component @main(%in: i32, %go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out: i32, %done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    %m1.addr0, %m1.write_data, %m1.write_en, %m1.write_done, %m1.clk, %m1.read_data, %m1.read_en, %m1.read_done = calyx.seq_mem @m1 <[64] x 32> [6] : i6, i32, i1, i1, i1, i32, i1, i1

    calyx.wires {
      // CHECK: done = m1.write_done
      calyx.assign %done = %m1.write_done : i1
      // CHECK: m1.write_en = go
      calyx.assign %m1.write_en = %go : i1
      // CHECK: m1.write_data = in
      calyx.assign %m1.write_data = %in : i32
      // CHECK: out = m1.read_data
      calyx.assign %out = %m1.read_data : i32
    }
    calyx.control {}
  } {static = 1}
}
