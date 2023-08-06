// RUN: circt-translate --export-calyx --split-input-file --verify-diagnostics %s | FileCheck %s --strict-whitespace

module attributes {calyx.entrypoint = "A"} {
  // CHECK-LABEL: extern "test.v" {
  // CHECK: primitive prim(in: 32) -> (out: 32);
  // CHECK: }
  hw.module.extern @prim(%in: i32) -> (out: i32) attributes {filename = "test.v"}
  // CHECK-LABEL: extern "test.v" {
  // CHECK: primitive params[WIDTH](in: WIDTH) -> (out: WIDTH);
  // CHECK: }
  hw.module.extern @params<WIDTH: i32>(%in: !hw.int<#hw.param.decl.ref<"WIDTH">>) -> (out: !hw.int<#hw.param.decl.ref<"WIDTH">>) attributes {filename = "test.v"}

  // CHECK-LABEL: component A<"static"=1>(in_0: 32, in_1: 32, @go go: 1, @clk clk: 1, @reset reset: 1) -> (out_0: 32, out_1: 32, @done done: 1) {
  calyx.component @A(%in_0: i32, %in_1: i32, %go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out_0: i32, %out_1: i32, %done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    // CHECK: params_0 = params(32);
    %params.in, %params.out = calyx.primitive @params_0 of @params<WIDTH: i32 = 32> : i32, i32
    // CHECK: prim_0 = prim();
    %prim.in, %prim.out = calyx.primitive @prim_0 of @prim : i32, i32

    calyx.wires {
      // CHECK: done = 1'd1;
      calyx.assign %done = %c1_1 : i1
      // CHECK: params_0.in = in_0;
      calyx.assign %params.in = %in_0 : i32
      // CHECK: out_0 = params_0.out;
      calyx.assign %out_0 = %params.out : i32
      // CHECK: prim_0.in = in_1;
      calyx.assign %prim.in = %in_1 : i32
      // CHECK: out_1 = prim_0.out;
      calyx.assign %out_1 = %prim.out : i32
    }
    calyx.control {}
  } {static = 1}
}

// -----
module attributes {calyx.entrypoint = "A"} {
  // CHECK-LABEL: extern "test.v" {
  // CHECK: primitive params[WIDTH](in: WIDTH, @clk clk: 1, @go go: 1) -> (out: WIDTH, @done done: 1);
  // CHECK: }
  hw.module.extern @params<WIDTH: i32>(%in: !hw.int<#hw.param.decl.ref<"WIDTH">>, %clk: i1 {calyx.clk}, %go: i1 {calyx.go}) -> (out: !hw.int<#hw.param.decl.ref<"WIDTH">>, done: i1 {calyx.done}) attributes {filename = "test.v"}

  // CHECK-LABEL: component A<"static"=1>(in_0: 32, in_1: 32, @go go: 1, @clk clk: 1, @reset reset: 1) -> (out_0: 32, out_1: 32, @done done: 1) {
  calyx.component @A(%in_0: i32, %in_1: i32, %go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out_0: i32, %out_1: i32, %done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    // CHECK: params_0 = params(32);
    %params.in, %params.clk, %params.go, %params.out, %params.done = calyx.primitive @params_0 of @params<WIDTH: i32 = 32> : i32, i1, i1, i32, i1

    calyx.wires {
      // CHECK: done = 1'd1;
      calyx.assign %done = %c1_1 : i1
      // CHECK: params_0.in = in_0;
      calyx.assign %params.in = %in_0 : i32
      // CHECK: out_0 = params_0.out;
      calyx.assign %out_0 = %params.out : i32
    }
    calyx.control {}
  } {static = 1}
}
