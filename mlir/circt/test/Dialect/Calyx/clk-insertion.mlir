// RUN: circt-opt -pass-pipeline='builtin.module(calyx.component(calyx-clk-insertion,calyx-reset-insertion))' %s | FileCheck %s

module attributes {calyx.entrypoint = "main"} {
  calyx.component @A(%in: i8, %go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out: i8, %done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %c0.in, %c0.go, %c0.clk, %c0.reset, %c0.out, %c0.done = calyx.instance @c0 of @A : i8, i1, i1, i1, i8, i1
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i1, i1, i1, i1, i1, i1
// CHECK: calyx.wires {
// CHECK:   calyx.assign %c0.reset = %reset : i1
// CHECK:   calyx.assign %r.reset = %reset : i1
// CHECK:   calyx.assign %c0.clk = %clk : i1
// CHECK:   calyx.assign %r.clk = %clk : i1
// CHECK: }
    calyx.wires {
    }
    calyx.control {
      calyx.seq { }
    }
  }
}
