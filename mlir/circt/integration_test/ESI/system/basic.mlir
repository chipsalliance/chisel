// REQUIRES: rtl-sim
// RUN: circt-opt %s --lower-esi-to-physical --lower-esi-ports --lower-esi-to-hw -verify-diagnostics > %t1.mlir
// RUN: circt-opt %t1.mlir -export-verilog -verify-diagnostics -o t3.mlir > %t2.sv
// RUN: circt-rtl-sim.py %t2.sv %BININC%/circt/Dialect/ESI/ESIPrimitives.sv %S/../supplements/integers.sv --cycles 150 | FileCheck %s

hw.module.extern @IntCountProd(%clk: i1, %rst: i1) -> (ints: !esi.channel<i32>) attributes {esi.bundle}
hw.module.extern @IntAcc(%clk: i1, %rst: i1, %ints: !esi.channel<i32>) -> (totalOut: i32) attributes {esi.bundle}
hw.module @top(%clk: i1, %rst: i1) -> (totalOut: i32) {
  %intStream = hw.instance "prod" @IntCountProd(clk: %clk: i1, rst: %rst: i1) -> (ints: !esi.channel<i32>)
  %intStreamBuffered = esi.buffer %clk, %rst, %intStream {stages=2, name="intChan"} : i32
  %totalOut = hw.instance "acc" @IntAcc(clk: %clk: i1, rst: %rst: i1, ints: %intStreamBuffered: !esi.channel<i32>) -> (totalOut: i32)
  hw.output %totalOut : i32
}
// CHECK:      [driver] Starting simulation
// CHECK: Total:          0
// CHECK: Data:     1
// CHECK: Total:          1
// CHECK: Data:     2
// CHECK: Total:          3
// CHECK: Data:     3
// CHECK: Total:          6
// CHECK: Data:     4
// CHECK: Total:         10
// CHECK: Data:     5
// CHECK: Total:         15
// CHECK: Data:     6
// CHECK: Total:         21
// CHECK: Data:     7
// CHECK: Total:         28
// CHECK: Data:     8
// CHECK: Total:         36
// CHECK: Data:     9
// CHECK: Total:         45
// CHECK: Data:    10
// CHECK: Total:         55
// CHECK: Data:    11
// CHECK: Total:         66
// CHECK: Data:    12
// CHECK: Total:         78
// CHECK: Data:    13
// CHECK: Total:         91
// CHECK: Data:    14
// CHECK: Total:        105
// CHECK: Data:    15
// CHECK: Total:        120
// CHECK: Data:    16
// CHECK: Total:        136
// CHECK: Data:    17
// CHECK: Total:        153
// CHECK: Data:    18
// CHECK: Total:        171
// CHECK: Data:    19
// CHECK: Total:        190
// CHECK: Data:    20
// CHECK: Total:        210
// CHECK: Data:    21
// CHECK: Total:        231
// CHECK: Data:    22
