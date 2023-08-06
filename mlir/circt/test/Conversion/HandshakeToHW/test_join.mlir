// RUN: circt-opt -lower-handshake-to-hw -split-input-file %s | FileCheck %s

// CHECK-LABEL:   hw.module @handshake_join_2ins_1outs_ctrl(
// CHECK-SAME:                                              %[[VAL_0:.*]]: !esi.channel<i0>,
// CHECK-SAME:                                              %[[VAL_1:.*]]: !esi.channel<i0>) -> (out0: !esi.channel<i0>) {
// CHECK:           %[[VAL_2:.*]], %[[VAL_3:.*]] = esi.unwrap.vr %[[VAL_0]], %[[VAL_4:.*]] : i0
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = esi.unwrap.vr %[[VAL_1]], %[[VAL_4]] : i0
// CHECK:           %[[VAL_7:.*]], %[[VAL_8:.*]] = esi.wrap.vr %[[VAL_9:.*]], %[[VAL_10:.*]] : i0
// CHECK:           %[[VAL_10]] = comb.and %[[VAL_3]], %[[VAL_6]] : i1
// CHECK:           %[[VAL_4]] = comb.and %[[VAL_8]], %[[VAL_10]] : i1
// CHECK:           %[[VAL_9]] = hw.constant 0 : i0
// CHECK:           hw.output %[[VAL_7]] : !esi.channel<i0>
// CHECK:         }

handshake.func @test_join(%arg0: none, %arg1: none, %arg2: none, ...) -> (none, none) {
  %0 = join %arg0, %arg1 : none, none
  return %0, %arg2 : none, none
}

// -----

// CHECK-LABEL:   hw.module @handshake_join_in_ui32_ui1_3ins_1outs_ctrl(
// CHECK-SAME:                                                          %[[VAL_0:.*]]: !esi.channel<i32>,
// CHECK-SAME:                                                          %[[VAL_1:.*]]: !esi.channel<i1>,
// CHECK-SAME:                                                          %[[VAL_2:.*]]: !esi.channel<i0>) -> (out0: !esi.channel<i0>) {
// CHECK:           %[[VAL_3:.*]], %[[VAL_4:.*]] = esi.unwrap.vr %[[VAL_0]], %[[VAL_5:.*]] : i32
// CHECK:           %[[VAL_6:.*]], %[[VAL_7:.*]] = esi.unwrap.vr %[[VAL_1]], %[[VAL_5]] : i1
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]] = esi.unwrap.vr %[[VAL_2]], %[[VAL_5]] : i0
// CHECK:           %[[VAL_10:.*]], %[[VAL_11:.*]] = esi.wrap.vr %[[VAL_12:.*]], %[[VAL_13:.*]] : i0
// CHECK:           %[[VAL_13]] = comb.and %[[VAL_4]], %[[VAL_7]], %[[VAL_9]] : i1
// CHECK:           %[[VAL_5]] = comb.and %[[VAL_11]], %[[VAL_13]] : i1
// CHECK:           %[[VAL_12]] = hw.constant 0 : i0
// CHECK:           hw.output %[[VAL_10]] : !esi.channel<i0>
// CHECK:         }

handshake.func @test_join_multi_types(%arg0: i32, %arg1: i1, %arg2: none, ...) -> (none) {
  %0 = join %arg0, %arg1, %arg2 : i32, i1, none
  return %0: none
}
