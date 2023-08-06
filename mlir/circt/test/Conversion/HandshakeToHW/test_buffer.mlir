// RUN: circt-opt -lower-handshake-to-hw --split-input-file %s | FileCheck %s

// CHECK-LABEL:   hw.module @handshake_buffer_3slots_seq_1ins_1outs_ctrl(
// CHECK-SAME:          %[[VAL_0:.*]]: !esi.channel<i0>, %[[VAL_1:.*]]: i1, %[[VAL_2:.*]]: i1) -> (out0: !esi.channel<i0>) {
// CHECK:           %[[VAL_3:.*]], %[[VAL_4:.*]] = esi.unwrap.vr %[[VAL_0]], %[[VAL_5:.*]] : i0
// CHECK:           %[[VAL_6:.*]], %[[VAL_7:.*]] = esi.wrap.vr %[[VAL_8:.*]], %[[VAL_9:.*]] : i0
// CHECK:           %[[VAL_10:.*]] = hw.constant 0 : i0
// CHECK:           %[[VAL_11:.*]] = hw.constant false
// CHECK:           %[[VAL_12:.*]] = seq.compreg %[[VAL_13:.*]], %[[VAL_1]], %[[VAL_2]], %[[VAL_11]]  : i1
// CHECK:           %[[VAL_14:.*]] = hw.constant true
// CHECK:           %[[VAL_15:.*]] = comb.xor %[[VAL_12]], %[[VAL_14]] : i1
// CHECK:           %[[VAL_5]] = comb.or %[[VAL_15]], %[[VAL_16:.*]] : i1
// CHECK:           %[[VAL_13]] = comb.mux %[[VAL_5]], %[[VAL_4]], %[[VAL_12]] : i1
// CHECK:           %[[VAL_17:.*]] = comb.mux %[[VAL_5]], %[[VAL_3]], %[[VAL_18:.*]] : i0
// CHECK:           %[[VAL_18]] = seq.compreg %[[VAL_17]], %[[VAL_1]], %[[VAL_2]], %[[VAL_10]]  : i0
// CHECK:           %[[VAL_19:.*]] = seq.compreg %[[VAL_20:.*]], %[[VAL_1]], %[[VAL_2]], %[[VAL_11]]  : i1
// CHECK:           %[[VAL_21:.*]] = comb.mux %[[VAL_19]], %[[VAL_19]], %[[VAL_12]] : i1
// CHECK:           %[[VAL_16]] = comb.xor %[[VAL_19]], %[[VAL_14]] : i1
// CHECK:           %[[VAL_22:.*]] = comb.xor %[[VAL_23:.*]], %[[VAL_14]] : i1
// CHECK:           %[[VAL_24:.*]] = comb.and %[[VAL_22]], %[[VAL_16]] : i1
// CHECK:           %[[VAL_25:.*]] = comb.mux %[[VAL_24]], %[[VAL_12]], %[[VAL_19]] : i1
// CHECK:           %[[VAL_26:.*]] = comb.and %[[VAL_23]], %[[VAL_19]] : i1
// CHECK:           %[[VAL_20]] = comb.mux %[[VAL_26]], %[[VAL_11]], %[[VAL_25]] : i1
// CHECK:           %[[VAL_27:.*]] = seq.compreg %[[VAL_28:.*]], %[[VAL_1]], %[[VAL_2]], %[[VAL_10]]  : i0
// CHECK:           %[[VAL_29:.*]] = comb.mux %[[VAL_19]], %[[VAL_27]], %[[VAL_18]] : i0
// CHECK:           %[[VAL_30:.*]] = comb.mux %[[VAL_24]], %[[VAL_18]], %[[VAL_27]] : i0
// CHECK:           %[[VAL_28]] = comb.mux %[[VAL_26]], %[[VAL_10]], %[[VAL_30]] : i0
// CHECK:           %[[VAL_31:.*]] = hw.constant 0 : i0
// CHECK:           %[[VAL_32:.*]] = seq.compreg %[[VAL_33:.*]], %[[VAL_1]], %[[VAL_2]], %[[VAL_11]]  : i1
// CHECK:           %[[VAL_34:.*]] = comb.xor %[[VAL_32]], %[[VAL_14]] : i1
// CHECK:           %[[VAL_23]] = comb.or %[[VAL_34]], %[[VAL_35:.*]] : i1
// CHECK:           %[[VAL_33]] = comb.mux %[[VAL_23]], %[[VAL_21]], %[[VAL_32]] : i1
// CHECK:           %[[VAL_36:.*]] = comb.mux %[[VAL_23]], %[[VAL_29]], %[[VAL_37:.*]] : i0
// CHECK:           %[[VAL_37]] = seq.compreg %[[VAL_36]], %[[VAL_1]], %[[VAL_2]], %[[VAL_31]]  : i0
// CHECK:           %[[VAL_38:.*]] = seq.compreg %[[VAL_39:.*]], %[[VAL_1]], %[[VAL_2]], %[[VAL_11]]  : i1
// CHECK:           %[[VAL_40:.*]] = comb.mux %[[VAL_38]], %[[VAL_38]], %[[VAL_32]] : i1
// CHECK:           %[[VAL_35]] = comb.xor %[[VAL_38]], %[[VAL_14]] : i1
// CHECK:           %[[VAL_41:.*]] = comb.xor %[[VAL_42:.*]], %[[VAL_14]] : i1
// CHECK:           %[[VAL_43:.*]] = comb.and %[[VAL_41]], %[[VAL_35]] : i1
// CHECK:           %[[VAL_44:.*]] = comb.mux %[[VAL_43]], %[[VAL_32]], %[[VAL_38]] : i1
// CHECK:           %[[VAL_45:.*]] = comb.and %[[VAL_42]], %[[VAL_38]] : i1
// CHECK:           %[[VAL_39]] = comb.mux %[[VAL_45]], %[[VAL_11]], %[[VAL_44]] : i1
// CHECK:           %[[VAL_46:.*]] = seq.compreg %[[VAL_47:.*]], %[[VAL_1]], %[[VAL_2]], %[[VAL_31]]  : i0
// CHECK:           %[[VAL_48:.*]] = comb.mux %[[VAL_38]], %[[VAL_46]], %[[VAL_37]] : i0
// CHECK:           %[[VAL_49:.*]] = comb.mux %[[VAL_43]], %[[VAL_37]], %[[VAL_46]] : i0
// CHECK:           %[[VAL_47]] = comb.mux %[[VAL_45]], %[[VAL_31]], %[[VAL_49]] : i0
// CHECK:           %[[VAL_50:.*]] = hw.constant 0 : i0
// CHECK:           %[[VAL_51:.*]] = seq.compreg %[[VAL_52:.*]], %[[VAL_1]], %[[VAL_2]], %[[VAL_11]]  : i1
// CHECK:           %[[VAL_53:.*]] = comb.xor %[[VAL_51]], %[[VAL_14]] : i1
// CHECK:           %[[VAL_42]] = comb.or %[[VAL_53]], %[[VAL_54:.*]] : i1
// CHECK:           %[[VAL_52]] = comb.mux %[[VAL_42]], %[[VAL_40]], %[[VAL_51]] : i1
// CHECK:           %[[VAL_55:.*]] = comb.mux %[[VAL_42]], %[[VAL_48]], %[[VAL_56:.*]] : i0
// CHECK:           %[[VAL_56]] = seq.compreg %[[VAL_55]], %[[VAL_1]], %[[VAL_2]], %[[VAL_50]]  : i0
// CHECK:           %[[VAL_57:.*]] = seq.compreg %[[VAL_58:.*]], %[[VAL_1]], %[[VAL_2]], %[[VAL_11]]  : i1
// CHECK:           %[[VAL_9]] = comb.mux %[[VAL_57]], %[[VAL_57]], %[[VAL_51]] : i1
// CHECK:           %[[VAL_54]] = comb.xor %[[VAL_57]], %[[VAL_14]] : i1
// CHECK:           %[[VAL_59:.*]] = comb.xor %[[VAL_7]], %[[VAL_14]] : i1
// CHECK:           %[[VAL_60:.*]] = comb.and %[[VAL_59]], %[[VAL_54]] : i1
// CHECK:           %[[VAL_61:.*]] = comb.mux %[[VAL_60]], %[[VAL_51]], %[[VAL_57]] : i1
// CHECK:           %[[VAL_62:.*]] = comb.and %[[VAL_7]], %[[VAL_57]] : i1
// CHECK:           %[[VAL_58]] = comb.mux %[[VAL_62]], %[[VAL_11]], %[[VAL_61]] : i1
// CHECK:           %[[VAL_63:.*]] = seq.compreg %[[VAL_64:.*]], %[[VAL_1]], %[[VAL_2]], %[[VAL_50]]  : i0
// CHECK:           %[[VAL_8]] = comb.mux %[[VAL_57]], %[[VAL_63]], %[[VAL_56]] : i0
// CHECK:           %[[VAL_65:.*]] = comb.mux %[[VAL_60]], %[[VAL_56]], %[[VAL_63]] : i0
// CHECK:           %[[VAL_64]] = comb.mux %[[VAL_62]], %[[VAL_50]], %[[VAL_65]] : i0
// CHECK:           hw.output %[[VAL_6]] : !esi.channel<i0>
// CHECK:         }

handshake.func @test_buffer_none(%arg0: none, %arg1: none, ...) -> (none, none) {
  %0 = buffer [3] seq %arg0 : none
  return %0, %arg1 : none, none
}

// -----

// CHECK-LABEL:   hw.module @handshake_buffer_in_ui64_out_ui64_2slots_seq(
// CHECK-SAME:           %[[VAL_0:.*]]: !esi.channel<i64>, %[[VAL_1:.*]]: i1, %[[VAL_2:.*]]: i1) -> (out0: !esi.channel<i64>) {
// CHECK:           %[[VAL_3:.*]], %[[VAL_4:.*]] = esi.unwrap.vr %[[VAL_0]], %[[VAL_5:.*]] : i64
// CHECK:           %[[VAL_6:.*]], %[[VAL_7:.*]] = esi.wrap.vr %[[VAL_8:.*]], %[[VAL_9:.*]] : i64
// CHECK:           %[[VAL_10:.*]] = hw.constant 0 : i64
// CHECK:           %[[VAL_11:.*]] = hw.constant false
// CHECK:           %[[VAL_12:.*]] = seq.compreg %[[VAL_13:.*]], %[[VAL_1]], %[[VAL_2]], %[[VAL_11]]  : i1
// CHECK:           %[[VAL_14:.*]] = hw.constant true
// CHECK:           %[[VAL_15:.*]] = comb.xor %[[VAL_12]], %[[VAL_14]] : i1
// CHECK:           %[[VAL_5]] = comb.or %[[VAL_15]], %[[VAL_16:.*]] : i1
// CHECK:           %[[VAL_13]] = comb.mux %[[VAL_5]], %[[VAL_4]], %[[VAL_12]] : i1
// CHECK:           %[[VAL_17:.*]] = comb.mux %[[VAL_5]], %[[VAL_3]], %[[VAL_18:.*]] : i64
// CHECK:           %[[VAL_18]] = seq.compreg %[[VAL_17]], %[[VAL_1]], %[[VAL_2]], %[[VAL_10]]  : i64
// CHECK:           %[[VAL_19:.*]] = seq.compreg %[[VAL_20:.*]], %[[VAL_1]], %[[VAL_2]], %[[VAL_11]]  : i1
// CHECK:           %[[VAL_21:.*]] = comb.mux %[[VAL_19]], %[[VAL_19]], %[[VAL_12]] : i1
// CHECK:           %[[VAL_16]] = comb.xor %[[VAL_19]], %[[VAL_14]] : i1
// CHECK:           %[[VAL_22:.*]] = comb.xor %[[VAL_23:.*]], %[[VAL_14]] : i1
// CHECK:           %[[VAL_24:.*]] = comb.and %[[VAL_22]], %[[VAL_16]] : i1
// CHECK:           %[[VAL_25:.*]] = comb.mux %[[VAL_24]], %[[VAL_12]], %[[VAL_19]] : i1
// CHECK:           %[[VAL_26:.*]] = comb.and %[[VAL_23]], %[[VAL_19]] : i1
// CHECK:           %[[VAL_20]] = comb.mux %[[VAL_26]], %[[VAL_11]], %[[VAL_25]] : i1
// CHECK:           %[[VAL_27:.*]] = seq.compreg %[[VAL_28:.*]], %[[VAL_1]], %[[VAL_2]], %[[VAL_10]]  : i64
// CHECK:           %[[VAL_29:.*]] = comb.mux %[[VAL_19]], %[[VAL_27]], %[[VAL_18]] : i64
// CHECK:           %[[VAL_30:.*]] = comb.mux %[[VAL_24]], %[[VAL_18]], %[[VAL_27]] : i64
// CHECK:           %[[VAL_28]] = comb.mux %[[VAL_26]], %[[VAL_10]], %[[VAL_30]] : i64
// CHECK:           %[[VAL_31:.*]] = seq.compreg %[[VAL_32:.*]], %[[VAL_1]], %[[VAL_2]], %[[VAL_11]]  : i1
// CHECK:           %[[VAL_33:.*]] = comb.xor %[[VAL_31]], %[[VAL_14]] : i1
// CHECK:           %[[VAL_23]] = comb.or %[[VAL_33]], %[[VAL_34:.*]] : i1
// CHECK:           %[[VAL_32]] = comb.mux %[[VAL_23]], %[[VAL_21]], %[[VAL_31]] : i1
// CHECK:           %[[VAL_35:.*]] = comb.mux %[[VAL_23]], %[[VAL_29]], %[[VAL_36:.*]] : i64
// CHECK:           %[[VAL_36]] = seq.compreg %[[VAL_35]], %[[VAL_1]], %[[VAL_2]], %[[VAL_10]]  : i64
// CHECK:           %[[VAL_37:.*]] = seq.compreg %[[VAL_38:.*]], %[[VAL_1]], %[[VAL_2]], %[[VAL_11]]  : i1
// CHECK:           %[[VAL_9]] = comb.mux %[[VAL_37]], %[[VAL_37]], %[[VAL_31]] : i1
// CHECK:           %[[VAL_34]] = comb.xor %[[VAL_37]], %[[VAL_14]] : i1
// CHECK:           %[[VAL_39:.*]] = comb.xor %[[VAL_7]], %[[VAL_14]] : i1
// CHECK:           %[[VAL_40:.*]] = comb.and %[[VAL_39]], %[[VAL_34]] : i1
// CHECK:           %[[VAL_41:.*]] = comb.mux %[[VAL_40]], %[[VAL_31]], %[[VAL_37]] : i1
// CHECK:           %[[VAL_42:.*]] = comb.and %[[VAL_7]], %[[VAL_37]] : i1
// CHECK:           %[[VAL_38]] = comb.mux %[[VAL_42]], %[[VAL_11]], %[[VAL_41]] : i1
// CHECK:           %[[VAL_43:.*]] = seq.compreg %[[VAL_44:.*]], %[[VAL_1]], %[[VAL_2]], %[[VAL_10]]  : i64
// CHECK:           %[[VAL_8]] = comb.mux %[[VAL_37]], %[[VAL_43]], %[[VAL_36]] : i64
// CHECK:           %[[VAL_45:.*]] = comb.mux %[[VAL_40]], %[[VAL_36]], %[[VAL_43]] : i64
// CHECK:           %[[VAL_44]] = comb.mux %[[VAL_42]], %[[VAL_10]], %[[VAL_45]] : i64
// CHECK:           hw.output %[[VAL_6]] : !esi.channel<i64>
// CHECK:         }

handshake.func @test_buffer_data(%arg0: index, %arg1: none, ...) -> (index, none) {
  %0 = buffer [2] seq %arg0 : index
  return %0, %arg1 : index, none
}

// -----

// CHECK-LABEL: hw.module @handshake_buffer_in_tuple_ui32_ui32_out_tuple_ui32_ui32_2slots_seq(%in0: !esi.channel<!hw.struct<field0: i32, field1: i32>>, %clock: i1, %reset: i1) -> (out0: !esi.channel<!hw.struct<field0: i32, field1: i32>>) {
// CHECK:         %[[CZERO:.*]] = hw.struct_create (%c0_i32, %c0_i32) : !hw.struct<field0: i32, field1: i32>
// CHECK:         %data0_reg = seq.compreg %4, %clock, %reset, %[[CZERO]]  : !hw.struct<field0: i32, field1: i32>

handshake.func @test_buffer_tuple_seq(%t: tuple<i32, i32>, %arg0: none, ...) -> (tuple<i32, i32>, none) {
  %0 = buffer [2] seq %t : tuple<i32, i32>
  return %0, %arg0 : tuple<i32, i32>, none
}
