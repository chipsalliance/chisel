// RUN: circt-opt -lower-handshake-to-hw -split-input-file %s | FileCheck %s

// CHECK-LABEL:   hw.module @handshake_constant_c42_out_ui64(
// CHECK-SAME:                                               %[[VAL_0:.*]]: !esi.channel<i0>) -> (out0: !esi.channel<i64>) {
// CHECK:           %[[VAL_1:.*]], %[[VAL_2:.*]] = esi.unwrap.vr %[[VAL_0]], %[[VAL_3:.*]] : i0
// CHECK:           %[[VAL_4:.*]], %[[VAL_3]] = esi.wrap.vr %[[VAL_5:.*]], %[[VAL_2]] : i64
// CHECK:           %[[VAL_5]] = hw.constant 42 : i64
// CHECK:           hw.output %[[VAL_4]] : !esi.channel<i64>
// CHECK:         }

// CHECK-LABEL:   hw.module @handshake_constant_c42_out_ui32(
// CHECK-SAME:                                               %[[VAL_0:.*]]: !esi.channel<i0>) -> (out0: !esi.channel<i32>) {
// CHECK:           %[[VAL_1:.*]], %[[VAL_2:.*]] = esi.unwrap.vr %[[VAL_0]], %[[VAL_3:.*]] : i0
// CHECK:           %[[VAL_4:.*]], %[[VAL_3]] = esi.wrap.vr %[[VAL_5:.*]], %[[VAL_2]] : i32
// CHECK:           %[[VAL_5]] = hw.constant 42 : i32
// CHECK:           hw.output %[[VAL_4]] : !esi.channel<i32>
// CHECK:         }

handshake.func @test_constant(%arg0: none, ...) -> (index, i32) {
  %0:2 = fork [2] %arg0 : none
  %1 = constant %0#0 {value = 42 : index} : index
  %2 = constant %0#1 {value = 42 : i32} : i32
  return %1, %2 : index, i32
}
