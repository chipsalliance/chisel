// RUN: circt-opt -lower-handshake-to-hw -split-input-file %s | FileCheck %s

// Test a control merge that is control only.

// CHECK-LABEL:   hw.module @handshake_control_merge_out_ui64_2ins_2outs_ctrl(
// CHECK-SAME:              %[[VAL_0:.*]]: !esi.channel<i0>, %[[VAL_1:.*]]: !esi.channel<i0>, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1) -> (dataOut: !esi.channel<i0>, index: !esi.channel<i64>) {
// CHECK:           %[[VAL_4:.*]], %[[VAL_5:.*]] = esi.unwrap.vr %[[VAL_0]], %[[VAL_6:.*]] : i0
// CHECK:           %[[VAL_7:.*]], %[[VAL_8:.*]] = esi.unwrap.vr %[[VAL_1]], %[[VAL_9:.*]] : i0
// CHECK:           %[[VAL_10:.*]], %[[VAL_11:.*]] = esi.wrap.vr %[[VAL_12:.*]], %[[VAL_13:.*]] : i0
// CHECK:           %[[VAL_14:.*]], %[[VAL_15:.*]] = esi.wrap.vr %[[VAL_16:.*]], %[[VAL_17:.*]] : i64
// CHECK:           %[[VAL_18:.*]] = hw.constant 0 : i2
// CHECK:           %[[VAL_19:.*]] = hw.constant false
// CHECK:           %[[VAL_20:.*]] = seq.compreg %[[VAL_21:.*]], %[[VAL_2]], %[[VAL_3]], %[[VAL_18]]  : i2
// CHECK:           %[[VAL_22:.*]] = seq.compreg %[[VAL_23:.*]], %[[VAL_2]], %[[VAL_3]], %[[VAL_19]]  : i1
// CHECK:           %[[VAL_24:.*]] = seq.compreg %[[VAL_25:.*]], %[[VAL_2]], %[[VAL_3]], %[[VAL_19]]  : i1
// CHECK:           %[[VAL_26:.*]] = comb.extract %[[VAL_27:.*]] from 0 : (i2) -> i1
// CHECK:           %[[VAL_28:.*]] = comb.extract %[[VAL_27]] from 1 : (i2) -> i1
// CHECK:           %[[VAL_29:.*]] = comb.or %[[VAL_26]], %[[VAL_28]] : i1
// CHECK:           %[[VAL_30:.*]] = comb.extract %[[VAL_20]] from 0 : (i2) -> i1
// CHECK:           %[[VAL_31:.*]] = comb.extract %[[VAL_20]] from 1 : (i2) -> i1
// CHECK:           %[[VAL_32:.*]] = comb.or %[[VAL_30]], %[[VAL_31]] : i1
// CHECK:           %[[VAL_33:.*]] = hw.constant -2 : i2
// CHECK:           %[[VAL_34:.*]] = comb.mux %[[VAL_8]], %[[VAL_33]], %[[VAL_18]] : i2
// CHECK:           %[[VAL_35:.*]] = hw.constant 1 : i2
// CHECK:           %[[VAL_36:.*]] = comb.mux %[[VAL_5]], %[[VAL_35]], %[[VAL_34]] : i2
// CHECK:           %[[VAL_27]] = comb.mux %[[VAL_32]], %[[VAL_20]], %[[VAL_36]] : i2
// CHECK:           %[[VAL_37:.*]] = hw.constant true
// CHECK:           %[[VAL_38:.*]] = comb.xor %[[VAL_22]], %[[VAL_37]] : i1
// CHECK:           %[[VAL_13]] = comb.and %[[VAL_29]], %[[VAL_38]] : i1
// CHECK:           %[[VAL_39:.*]] = hw.constant 0 : i0
// CHECK:           %[[VAL_40:.*]] = comb.extract %[[VAL_27]] from 1 : (i2) -> i1
// CHECK:           %[[VAL_12]] = comb.mux %[[VAL_40]], %[[VAL_7]], %[[VAL_39]] : i0
// CHECK:           %[[VAL_41:.*]] = comb.xor %[[VAL_24]], %[[VAL_37]] : i1
// CHECK:           %[[VAL_17]] = comb.and %[[VAL_29]], %[[VAL_41]] : i1
// CHECK:           %[[VAL_42:.*]] = hw.constant 0 : i64
// CHECK:           %[[VAL_43:.*]] = hw.constant 1 : i64
// CHECK:           %[[VAL_44:.*]] = comb.extract %[[VAL_27]] from 1 : (i2) -> i1
// CHECK:           %[[VAL_16]] = comb.mux %[[VAL_44]], %[[VAL_43]], %[[VAL_42]] : i64
// CHECK:           %[[VAL_21]] = comb.mux %[[VAL_45:.*]], %[[VAL_18]], %[[VAL_27]] : i2
// CHECK:           %[[VAL_46:.*]] = comb.and %[[VAL_13]], %[[VAL_11]] : i1
// CHECK:           %[[VAL_47:.*]] = comb.or %[[VAL_46]], %[[VAL_22]] : i1
// CHECK:           %[[VAL_48:.*]] = comb.and %[[VAL_17]], %[[VAL_15]] : i1
// CHECK:           %[[VAL_49:.*]] = comb.or %[[VAL_48]], %[[VAL_24]] : i1
// CHECK:           %[[VAL_45]] = comb.and %[[VAL_47]], %[[VAL_49]] : i1
// CHECK:           %[[VAL_23]] = comb.mux %[[VAL_45]], %[[VAL_19]], %[[VAL_47]] : i1
// CHECK:           %[[VAL_25]] = comb.mux %[[VAL_45]], %[[VAL_19]], %[[VAL_49]] : i1
// CHECK:           %[[VAL_50:.*]] = comb.mux %[[VAL_45]], %[[VAL_27]], %[[VAL_18]] : i2
// CHECK:           %[[VAL_6]] = comb.icmp eq %[[VAL_50]], %[[VAL_35]] : i2
// CHECK:           %[[VAL_9]] = comb.icmp eq %[[VAL_50]], %[[VAL_33]] : i2
// CHECK:           hw.output %[[VAL_10]], %[[VAL_14]] : !esi.channel<i0>, !esi.channel<i64>
// CHECK:         }

handshake.func @test_cmerge(%arg0: none, %arg1: none, %arg2: none, ...) -> (none, index, none) {
  %0:2 = control_merge %arg0, %arg1 : none, index
  return %0#0, %0#1, %arg2 : none, index, none
}

// -----

// Test a control merge that also outputs the selected input's data.

// CHECK-LABEL:   hw.module @handshake_control_merge_in_ui64_ui64_ui64_out_ui64_ui64(
// CHECK-SAME:                 %[[VAL_0:.*]]: !esi.channel<i64>, %[[VAL_1:.*]]: !esi.channel<i64>, %[[VAL_2:.*]]: !esi.channel<i64>, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (dataOut: !esi.channel<i64>, index: !esi.channel<i64>) {
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = esi.unwrap.vr %[[VAL_0]], %[[VAL_7:.*]] : i64
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]] = esi.unwrap.vr %[[VAL_1]], %[[VAL_10:.*]] : i64
// CHECK:           %[[VAL_11:.*]], %[[VAL_12:.*]] = esi.unwrap.vr %[[VAL_2]], %[[VAL_13:.*]] : i64
// CHECK:           %[[VAL_14:.*]], %[[VAL_15:.*]] = esi.wrap.vr %[[VAL_16:.*]], %[[VAL_17:.*]] : i64
// CHECK:           %[[VAL_18:.*]], %[[VAL_19:.*]] = esi.wrap.vr %[[VAL_20:.*]], %[[VAL_21:.*]] : i64
// CHECK:           %[[VAL_22:.*]] = hw.constant 0 : i3
// CHECK:           %[[VAL_23:.*]] = hw.constant false
// CHECK:           %[[VAL_24:.*]] = seq.compreg %[[VAL_25:.*]], %[[VAL_3]], %[[VAL_4]], %[[VAL_22]]  : i3
// CHECK:           %[[VAL_26:.*]] = seq.compreg %[[VAL_27:.*]], %[[VAL_3]], %[[VAL_4]], %[[VAL_23]]  : i1
// CHECK:           %[[VAL_28:.*]] = seq.compreg %[[VAL_29:.*]], %[[VAL_3]], %[[VAL_4]], %[[VAL_23]]  : i1
// CHECK:           %[[VAL_30:.*]] = comb.extract %[[VAL_31:.*]] from 0 : (i3) -> i1
// CHECK:           %[[VAL_32:.*]] = comb.extract %[[VAL_31]] from 1 : (i3) -> i1
// CHECK:           %[[VAL_33:.*]] = comb.extract %[[VAL_31]] from 2 : (i3) -> i1
// CHECK:           %[[VAL_34:.*]] = comb.or %[[VAL_30]], %[[VAL_32]], %[[VAL_33]] : i1
// CHECK:           %[[VAL_35:.*]] = comb.extract %[[VAL_24]] from 0 : (i3) -> i1
// CHECK:           %[[VAL_36:.*]] = comb.extract %[[VAL_24]] from 1 : (i3) -> i1
// CHECK:           %[[VAL_37:.*]] = comb.extract %[[VAL_24]] from 2 : (i3) -> i1
// CHECK:           %[[VAL_38:.*]] = comb.or %[[VAL_35]], %[[VAL_36]], %[[VAL_37]] : i1
// CHECK:           %[[VAL_39:.*]] = hw.constant -4 : i3
// CHECK:           %[[VAL_40:.*]] = comb.mux %[[VAL_12]], %[[VAL_39]], %[[VAL_22]] : i3
// CHECK:           %[[VAL_41:.*]] = hw.constant 2 : i3
// CHECK:           %[[VAL_42:.*]] = comb.mux %[[VAL_9]], %[[VAL_41]], %[[VAL_40]] : i3
// CHECK:           %[[VAL_43:.*]] = hw.constant 1 : i3
// CHECK:           %[[VAL_44:.*]] = comb.mux %[[VAL_6]], %[[VAL_43]], %[[VAL_42]] : i3
// CHECK:           %[[VAL_31]] = comb.mux %[[VAL_38]], %[[VAL_24]], %[[VAL_44]] : i3
// CHECK:           %[[VAL_45:.*]] = hw.constant true
// CHECK:           %[[VAL_46:.*]] = comb.xor %[[VAL_26]], %[[VAL_45]] : i1
// CHECK:           %[[VAL_17]] = comb.and %[[VAL_34]], %[[VAL_46]] : i1
// CHECK:           %[[VAL_47:.*]] = hw.constant 0 : i64
// CHECK:           %[[VAL_48:.*]] = comb.extract %[[VAL_31]] from 2 : (i3) -> i1
// CHECK:           %[[VAL_49:.*]] = comb.mux %[[VAL_48]], %[[VAL_11]], %[[VAL_47]] : i64
// CHECK:           %[[VAL_50:.*]] = comb.extract %[[VAL_31]] from 1 : (i3) -> i1
// CHECK:           %[[VAL_16]] = comb.mux %[[VAL_50]], %[[VAL_8]], %[[VAL_49]] : i64
// CHECK:           %[[VAL_51:.*]] = comb.xor %[[VAL_28]], %[[VAL_45]] : i1
// CHECK:           %[[VAL_21]] = comb.and %[[VAL_34]], %[[VAL_51]] : i1
// CHECK:           %[[VAL_52:.*]] = hw.constant 1 : i64
// CHECK:           %[[VAL_53:.*]] = hw.constant 2 : i64
// CHECK:           %[[VAL_54:.*]] = comb.extract %[[VAL_31]] from 2 : (i3) -> i1
// CHECK:           %[[VAL_55:.*]] = comb.mux %[[VAL_54]], %[[VAL_53]], %[[VAL_47]] : i64
// CHECK:           %[[VAL_56:.*]] = comb.extract %[[VAL_31]] from 1 : (i3) -> i1
// CHECK:           %[[VAL_20]] = comb.mux %[[VAL_56]], %[[VAL_52]], %[[VAL_55]] : i64
// CHECK:           %[[VAL_25]] = comb.mux %[[VAL_57:.*]], %[[VAL_22]], %[[VAL_31]] : i3
// CHECK:           %[[VAL_58:.*]] = comb.and %[[VAL_17]], %[[VAL_15]] : i1
// CHECK:           %[[VAL_59:.*]] = comb.or %[[VAL_58]], %[[VAL_26]] : i1
// CHECK:           %[[VAL_60:.*]] = comb.and %[[VAL_21]], %[[VAL_19]] : i1
// CHECK:           %[[VAL_61:.*]] = comb.or %[[VAL_60]], %[[VAL_28]] : i1
// CHECK:           %[[VAL_57]] = comb.and %[[VAL_59]], %[[VAL_61]] : i1
// CHECK:           %[[VAL_27]] = comb.mux %[[VAL_57]], %[[VAL_23]], %[[VAL_59]] : i1
// CHECK:           %[[VAL_29]] = comb.mux %[[VAL_57]], %[[VAL_23]], %[[VAL_61]] : i1
// CHECK:           %[[VAL_62:.*]] = comb.mux %[[VAL_57]], %[[VAL_31]], %[[VAL_22]] : i3
// CHECK:           %[[VAL_7]] = comb.icmp eq %[[VAL_62]], %[[VAL_43]] : i3
// CHECK:           %[[VAL_10]] = comb.icmp eq %[[VAL_62]], %[[VAL_41]] : i3
// CHECK:           %[[VAL_13]] = comb.icmp eq %[[VAL_62]], %[[VAL_39]] : i3
// CHECK:           hw.output %[[VAL_14]], %[[VAL_18]] : !esi.channel<i64>, !esi.channel<i64>
// CHECK:         }

handshake.func @test_cmerge_data(%arg0: index, %arg1: index, %arg2: index) -> (index, index) {
  %0:2 = control_merge %arg0, %arg1, %arg2 : index, index
  return %0#0, %0#1 : index, index
}
