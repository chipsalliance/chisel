// RUN: circt-opt -lower-handshake-to-hw -split-input-file %s | FileCheck %s

// CHECK-LABEL:   hw.module @handshake_fork_1ins_2outs_ctrl(
// CHECK-SAME:            %[[VAL_0:.*]]: !esi.channel<i0>, %[[VAL_1:.*]]: i1, %[[VAL_2:.*]]: i1) -> (out0: !esi.channel<i0>, out1: !esi.channel<i0>) {
// CHECK:           %[[VAL_3:.*]], %[[VAL_4:.*]] = esi.unwrap.vr %[[VAL_0]], %[[VAL_5:.*]] : i0
// CHECK:           %[[VAL_6:.*]], %[[VAL_7:.*]] = esi.wrap.vr %[[VAL_3]], %[[VAL_8:.*]] : i0
// CHECK:           %[[VAL_9:.*]], %[[VAL_10:.*]] = esi.wrap.vr %[[VAL_3]], %[[VAL_11:.*]] : i0
// CHECK:           %[[VAL_12:.*]] = hw.constant false
// CHECK:           %[[VAL_13:.*]] = hw.constant true
// CHECK:           %[[VAL_14:.*]] = comb.xor %[[VAL_5]], %[[VAL_13]] : i1
// CHECK:           %[[VAL_15:.*]] = comb.and %[[VAL_16:.*]], %[[VAL_14]] : i1
// CHECK:           %[[VAL_17:.*]] = seq.compreg %[[VAL_15]], %[[VAL_1]], %[[VAL_2]], %[[VAL_12]]  : i1
// CHECK:           %[[VAL_18:.*]] = comb.xor %[[VAL_17]], %[[VAL_13]] : i1
// CHECK:           %[[VAL_8]] = comb.and %[[VAL_18]], %[[VAL_4]] : i1
// CHECK:           %[[VAL_19:.*]] = comb.and %[[VAL_7]], %[[VAL_8]] : i1
// CHECK:           %[[VAL_16]] = comb.or %[[VAL_19]], %[[VAL_17]] {sv.namehint = "done0"} : i1
// CHECK:           %[[VAL_20:.*]] = comb.xor %[[VAL_5]], %[[VAL_13]] : i1
// CHECK:           %[[VAL_21:.*]] = comb.and %[[VAL_22:.*]], %[[VAL_20]] : i1
// CHECK:           %[[VAL_23:.*]] = seq.compreg %[[VAL_21]], %[[VAL_1]], %[[VAL_2]], %[[VAL_12]]  : i1
// CHECK:           %[[VAL_24:.*]] = comb.xor %[[VAL_23]], %[[VAL_13]] : i1
// CHECK:           %[[VAL_11]] = comb.and %[[VAL_24]], %[[VAL_4]] : i1
// CHECK:           %[[VAL_25:.*]] = comb.and %[[VAL_10]], %[[VAL_11]] : i1
// CHECK:           %[[VAL_22]] = comb.or %[[VAL_25]], %[[VAL_23]] {sv.namehint = "done1"} : i1
// CHECK:           %[[VAL_5]] = comb.and %[[VAL_16]], %[[VAL_22]] {sv.namehint = "allDone"} : i1
// CHECK:           hw.output %[[VAL_6]], %[[VAL_9]] : !esi.channel<i0>, !esi.channel<i0>
// CHECK:         }

handshake.func @test_fork(%arg0: none, %arg1: none, ...) -> (none, none, none) {
  %0:2 = fork [2] %arg0 : none
  return %0#0, %0#1, %arg1 : none, none, none
}

// -----

// CHECK-LABEL:   hw.module @handshake_fork_in_ui64_out_ui64_ui64(
// CHECK-SAME:            %[[VAL_0:.*]]: !esi.channel<i64>, %[[VAL_1:.*]]: i1, %[[VAL_2:.*]]: i1) -> (out0: !esi.channel<i64>, out1: !esi.channel<i64>) {
// CHECK:           %[[VAL_3:.*]], %[[VAL_4:.*]] = esi.unwrap.vr %[[VAL_0]], %[[VAL_5:.*]] : i64
// CHECK:           %[[VAL_6:.*]], %[[VAL_7:.*]] = esi.wrap.vr %[[VAL_3]], %[[VAL_8:.*]] : i64
// CHECK:           %[[VAL_9:.*]], %[[VAL_10:.*]] = esi.wrap.vr %[[VAL_3]], %[[VAL_11:.*]] : i64
// CHECK:           %[[VAL_12:.*]] = hw.constant false
// CHECK:           %[[VAL_13:.*]] = hw.constant true
// CHECK:           %[[VAL_14:.*]] = comb.xor %[[VAL_5]], %[[VAL_13]] : i1
// CHECK:           %[[VAL_15:.*]] = comb.and %[[VAL_16:.*]], %[[VAL_14]] : i1
// CHECK:           %[[VAL_17:.*]] = seq.compreg %[[VAL_15]], %[[VAL_1]], %[[VAL_2]], %[[VAL_12]]  : i1
// CHECK:           %[[VAL_18:.*]] = comb.xor %[[VAL_17]], %[[VAL_13]] : i1
// CHECK:           %[[VAL_8]] = comb.and %[[VAL_18]], %[[VAL_4]] : i1
// CHECK:           %[[VAL_19:.*]] = comb.and %[[VAL_7]], %[[VAL_8]] : i1
// CHECK:           %[[VAL_16]] = comb.or %[[VAL_19]], %[[VAL_17]] {sv.namehint = "done0"} : i1
// CHECK:           %[[VAL_20:.*]] = comb.xor %[[VAL_5]], %[[VAL_13]] : i1
// CHECK:           %[[VAL_21:.*]] = comb.and %[[VAL_22:.*]], %[[VAL_20]] : i1
// CHECK:           %[[VAL_23:.*]] = seq.compreg %[[VAL_21]], %[[VAL_1]], %[[VAL_2]], %[[VAL_12]]  : i1
// CHECK:           %[[VAL_24:.*]] = comb.xor %[[VAL_23]], %[[VAL_13]] : i1
// CHECK:           %[[VAL_11]] = comb.and %[[VAL_24]], %[[VAL_4]] : i1
// CHECK:           %[[VAL_25:.*]] = comb.and %[[VAL_10]], %[[VAL_11]] : i1
// CHECK:           %[[VAL_22]] = comb.or %[[VAL_25]], %[[VAL_23]] {sv.namehint = "done1"} : i1
// CHECK:           %[[VAL_5]] = comb.and %[[VAL_16]], %[[VAL_22]] {sv.namehint = "allDone"} : i1
// CHECK:           hw.output %[[VAL_6]], %[[VAL_9]] : !esi.channel<i64>, !esi.channel<i64>
// CHECK:         }

handshake.func @test_fork_data(%arg0: index, %arg1: none, ...) -> (index, index, none) {
  %0:2 = fork [2] %arg0 : index
  return %0#0, %0#1, %arg1 : index, index, none
}
