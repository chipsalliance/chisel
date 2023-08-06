// RUN: circt-opt --split-input-file -lower-calyx-to-hw %s | FileCheck %s

// CHECK: hw.module @main(%in0: i4, %clk: i1, %reset: i1, %go: i1) -> (out0: i8, done: i1) {
// CHECK:   %out0 = sv.wire  : !hw.inout<i8>
// CHECK:   %0 = sv.read_inout %out0 : !hw.inout<i8>
// CHECK:   %done = sv.wire  : !hw.inout<i1>
// CHECK:   %1 = sv.read_inout %done : !hw.inout<i1>
// CHECK:   %true = hw.constant true
// CHECK:   %[[IN_WIRE:.*]] = sv.wire  : !hw.inout<i4>
// CHECK:   %[[IN_WIRE_READ:.*]] = sv.read_inout %[[IN_WIRE]] : !hw.inout<i4>
// CHECK:   %c0_i4 = hw.constant 0 : i4
// CHECK:   %[[PADDED:.*]] = comb.concat %c0_i4, %[[IN_WIRE_READ]] : i4, i4
// CHECK:   %[[PADDED_WIRE:.*]] = sv.wire  : !hw.inout<i8>
// CHECK:   sv.assign %[[PADDED_WIRE]], %[[PADDED]] : i8
// CHECK:   %[[PADDED_WIRE_READ:.*]] = sv.read_inout %[[PADDED_WIRE]] : !hw.inout<i8>
// CHECK:   sv.assign %[[IN_WIRE]], %in0 : i4
// CHECK:   sv.assign %out0, %[[PADDED_WIRE_READ]] : i8
// CHECK:   sv.assign %done, %true : i1
// CHECK:   hw.output %0, %1 : i8, i1
// CHECK: }
module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%in0: i4, %clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%out0: i8, %done: i1 {done}) {
    %true = hw.constant true
    %std_pad.in, %std_pad.out = calyx.std_pad @std_pad : i4, i8
    calyx.wires {
      calyx.assign %std_pad.in = %in0 : i4
      calyx.assign %out0 = %std_pad.out : i8
      calyx.assign %done = %true : i1
    }
    calyx.control {}
  }
}

// -----

// CHECK: hw.module @main(%in0: i4, %clk: i1, %reset: i1, %go: i1) -> (out0: i8, done: i1) {
// CHECK:   %out0 = sv.wire  : !hw.inout<i8>
// CHECK:   %0 = sv.read_inout %out0 : !hw.inout<i8>
// CHECK:   %done = sv.wire  : !hw.inout<i1>
// CHECK:   %1 = sv.read_inout %done : !hw.inout<i1>
// CHECK:   %true = hw.constant true
// CHECK:   %[[IN_WIRE:.*]] = sv.wire  : !hw.inout<i4>
// CHECK:   %[[IN_WIRE_READ:.*]] = sv.read_inout %[[IN_WIRE]] : !hw.inout<i4>
// CHECK:   %[[SIGNBIT:.*]] = comb.extract %[[IN_WIRE_READ]] from 3 : (i4) -> i1
// CHECK:   %[[SIGNBITVEC:.*]] = comb.replicate %[[SIGNBIT]] : (i1) -> i4
// CHECK:   %[[EXTENDED:.*]] = comb.concat %[[SIGNBITVEC]], %[[IN_WIRE_READ]] : i4, i4
// CHECK:   %[[EXTENDED_WIRE:.*]] = sv.wire  : !hw.inout<i8>
// CHECK:   sv.assign %[[EXTENDED_WIRE]], %[[EXTENDED]] : i8
// CHECK:   %[[EXTENDED_WIRE_READ:.*]] = sv.read_inout %[[EXTENDED_WIRE]] : !hw.inout<i8>
// CHECK:   sv.assign %[[IN_WIRE]], %in0 : i4
// CHECK:   sv.assign %out0, %[[EXTENDED_WIRE_READ]] : i8
// CHECK:   sv.assign %done, %true : i1
// CHECK:   hw.output %0, %1 : i8, i1

module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%in0: i4, %clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%out0: i8, %done: i1 {done}) {
    %true = hw.constant true
    %std_extsi.in, %std_extsi.out = calyx.std_extsi @std_extsi : i4, i8
    calyx.wires {
      calyx.assign %std_extsi.in = %in0 : i4
      calyx.assign %out0 = %std_extsi.out : i8
      calyx.assign %done = %true : i1
    }
    calyx.control {}
  }
}

// -----

// CHECK: hw.module @main(%in0: i8, %in1: i8, %cond0: i1, %cond1: i1, %clk: i1, %reset: i1, %go: i1) -> (out: i8, done: i1) {
// CHECK:   %out = sv.wire  : !hw.inout<i8>
// CHECK:   %0 = sv.read_inout %out : !hw.inout<i8>
// CHECK:   %done = sv.wire  : !hw.inout<i1>
// CHECK:   %1 = sv.read_inout %done : !hw.inout<i1>
// CHECK:   %true = hw.constant true
// CHECK:   %c0_i8 = hw.constant 0 : i8
// CHECK:   %[[MUX0:.*]] = comb.mux %cond0, %in0, %c0_i8 : i8
// CHECK:   %[[MUX1:.*]] = comb.mux %cond1, %in1, %[[MUX0]] : i8
// CHECK:   sv.assign %out, %[[MUX1]] : i8
// CHECK:   sv.assign %done, %true : i1
// CHECK:   hw.output %0, %1 : i8, i1
// CHECK: }
module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%in0: i8, %in1: i8, %cond0: i1, %cond1: i1, %clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%out: i8, %done: i1 {done}) {
    %true = hw.constant true
    calyx.wires {
      calyx.assign %out = %cond0 ? %in0 : i8
      calyx.assign %out = %cond1 ? %in1 : i8
      calyx.assign %done = %true : i1
    }
    calyx.control {}
  }
}
