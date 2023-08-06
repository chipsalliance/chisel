// RUN: circt-opt -lower-handshake-to-hw -split-input-file %s | FileCheck %s

// CHECK-LABEL:   hw.module @main(
// CHECK-SAME:                    %[[VAL_0:.*]]: !esi.channel<i0>,
// CHECK-SAME:                    %[[VAL_1:.*]]: i1,
// CHECK-SAME:                    %[[VAL_2:.*]]: i1) -> (out0: !esi.channel<i64>, outCtrl: !esi.channel<i0>) {
// CHECK:           %[[VAL_3:.*]], %[[VAL_4:.*]], %[[VAL_5:.*]], %[[VAL_6:.*]] = hw.instance "handshake_fork0" @handshake_fork_1ins_4outs_ctrl(in0: %[[VAL_0]]: !esi.channel<i0>, clock: %[[VAL_1]]: i1, reset: %[[VAL_2]]: i1) -> (out0: !esi.channel<i0>, out1: !esi.channel<i0>, out2: !esi.channel<i0>, out3: !esi.channel<i0>)
// CHECK:           %[[VAL_7:.*]] = hw.instance "handshake_constant0" @handshake_constant_c1_out_ui64(ctrl: %[[VAL_5]]: !esi.channel<i0>) -> (out0: !esi.channel<i64>)
// CHECK:           %[[VAL_8:.*]] = hw.instance "handshake_constant1" @handshake_constant_c42_out_ui64(ctrl: %[[VAL_4]]: !esi.channel<i0>) -> (out0: !esi.channel<i64>)
// CHECK:           %[[VAL_9:.*]] = hw.instance "handshake_constant2" @handshake_constant_c1_out_ui64(ctrl: %[[VAL_3]]: !esi.channel<i0>) -> (out0: !esi.channel<i64>)
// CHECK:           %[[VAL_10:.*]] = hw.instance "handshake_buffer0" @handshake_buffer_in_ui1_out_ui1_1slots_seq_init_0(in0: %[[VAL_11:.*]]: !esi.channel<i1>, clock: %[[VAL_1]]: i1, reset: %[[VAL_2]]: i1) -> (out0: !esi.channel<i1>)
// CHECK:           %[[VAL_12:.*]], %[[VAL_13:.*]], %[[VAL_14:.*]], %[[VAL_15:.*]] = hw.instance "handshake_fork1" @handshake_fork_in_ui1_out_ui1_ui1_ui1_ui1(in0: %[[VAL_10]]: !esi.channel<i1>, clock: %[[VAL_1]]: i1, reset: %[[VAL_2]]: i1) -> (out0: !esi.channel<i1>, out1: !esi.channel<i1>, out2: !esi.channel<i1>, out3: !esi.channel<i1>)
// CHECK:           %[[VAL_16:.*]] = hw.instance "handshake_mux0" @handshake_mux_in_ui1_3ins_1outs_ctrl(select: %[[VAL_15]]: !esi.channel<i1>, in0: %[[VAL_6]]: !esi.channel<i0>, in1: %[[VAL_17:.*]]: !esi.channel<i0>) -> (out0: !esi.channel<i0>)
// CHECK:           %[[VAL_18:.*]] = hw.instance "handshake_mux1" @handshake_mux_in_ui1_ui64_ui64_out_ui64(select: %[[VAL_14]]: !esi.channel<i1>, in0: %[[VAL_8]]: !esi.channel<i64>, in1: %[[VAL_19:.*]]: !esi.channel<i64>) -> (out0: !esi.channel<i64>)
// CHECK:           %[[VAL_20:.*]], %[[VAL_21:.*]] = hw.instance "handshake_fork2" @handshake_fork_in_ui64_out_ui64_ui64(in0: %[[VAL_18]]: !esi.channel<i64>, clock: %[[VAL_1]]: i1, reset: %[[VAL_2]]: i1) -> (out0: !esi.channel<i64>, out1: !esi.channel<i64>)
// CHECK:           %[[VAL_22:.*]] = hw.instance "handshake_mux2" @handshake_mux_in_ui1_ui64_ui64_out_ui64(select: %[[VAL_13]]: !esi.channel<i1>, in0: %[[VAL_9]]: !esi.channel<i64>, in1: %[[VAL_23:.*]]: !esi.channel<i64>) -> (out0: !esi.channel<i64>)
// CHECK:           %[[VAL_24:.*]] = hw.instance "handshake_mux3" @handshake_mux_in_ui1_ui64_ui64_out_ui64(select: %[[VAL_12]]: !esi.channel<i1>, in0: %[[VAL_7]]: !esi.channel<i64>, in1: %[[VAL_25:.*]]: !esi.channel<i64>) -> (out0: !esi.channel<i64>)
// CHECK:           %[[VAL_26:.*]], %[[VAL_27:.*]] = hw.instance "handshake_fork3" @handshake_fork_in_ui64_out_ui64_ui64(in0: %[[VAL_24]]: !esi.channel<i64>, clock: %[[VAL_1]]: i1, reset: %[[VAL_2]]: i1) -> (out0: !esi.channel<i64>, out1: !esi.channel<i64>)
// CHECK:           %[[VAL_28:.*]] = hw.instance "arith_cmpi0" @arith_cmpi_in_ui64_ui64_out_ui1_slt(in0: %[[VAL_26]]: !esi.channel<i64>, in1: %[[VAL_20]]: !esi.channel<i64>) -> (out0: !esi.channel<i1>)
// CHECK:           %[[VAL_11]], %[[VAL_29:.*]], %[[VAL_30:.*]], %[[VAL_31:.*]], %[[VAL_32:.*]] = hw.instance "handshake_fork4" @handshake_fork_in_ui1_out_ui1_ui1_ui1_ui1_ui1(in0: %[[VAL_28]]: !esi.channel<i1>, clock: %[[VAL_1]]: i1, reset: %[[VAL_2]]: i1) -> (out0: !esi.channel<i1>, out1: !esi.channel<i1>, out2: !esi.channel<i1>, out3: !esi.channel<i1>, out4: !esi.channel<i1>)
// CHECK:           %[[VAL_19]], %[[VAL_33:.*]] = hw.instance "handshake_cond_br0" @handshake_cond_br_in_ui1_ui64_out_ui64_ui64(cond: %[[VAL_32]]: !esi.channel<i1>, data: %[[VAL_21]]: !esi.channel<i64>) -> (outTrue: !esi.channel<i64>, outFalse: !esi.channel<i64>)
// CHECK:           hw.instance "handshake_sink0" @handshake_sink_in_ui64(in0: %[[VAL_33]]: !esi.channel<i64>) -> ()
// CHECK:           %[[VAL_34:.*]], %[[VAL_35:.*]] = hw.instance "handshake_cond_br1" @handshake_cond_br_in_ui1_ui64_out_ui64_ui64(cond: %[[VAL_31]]: !esi.channel<i1>, data: %[[VAL_22]]: !esi.channel<i64>) -> (outTrue: !esi.channel<i64>, outFalse: !esi.channel<i64>)
// CHECK:           hw.instance "handshake_sink1" @handshake_sink_in_ui64(in0: %[[VAL_35]]: !esi.channel<i64>) -> ()
// CHECK:           %[[VAL_17]], %[[VAL_36:.*]] = hw.instance "handshake_cond_br2" @handshake_cond_br_in_ui1_2ins_2outs_ctrl(cond: %[[VAL_30]]: !esi.channel<i1>, data: %[[VAL_16]]: !esi.channel<i0>) -> (outTrue: !esi.channel<i0>, outFalse: !esi.channel<i0>)
// CHECK:           %[[VAL_37:.*]], %[[VAL_38:.*]] = hw.instance "handshake_cond_br3" @handshake_cond_br_in_ui1_ui64_out_ui64_ui64(cond: %[[VAL_29]]: !esi.channel<i1>, data: %[[VAL_27]]: !esi.channel<i64>) -> (outTrue: !esi.channel<i64>, outFalse: !esi.channel<i64>)
// CHECK:           %[[VAL_23]], %[[VAL_39:.*]] = hw.instance "handshake_fork5" @handshake_fork_in_ui64_out_ui64_ui64(in0: %[[VAL_34]]: !esi.channel<i64>, clock: %[[VAL_1]]: i1, reset: %[[VAL_2]]: i1) -> (out0: !esi.channel<i64>, out1: !esi.channel<i64>)
// CHECK:           %[[VAL_25]] = hw.instance "arith_addi0" @arith_addi_in_ui64_ui64_out_ui64(in0: %[[VAL_37]]: !esi.channel<i64>, in1: %[[VAL_39]]: !esi.channel<i64>) -> (out0: !esi.channel<i64>)
// CHECK:           hw.output %[[VAL_38]], %[[VAL_36]] : !esi.channel<i64>, !esi.channel<i0>
// CHECK:         }

handshake.func @main(%arg0: none, ...) -> (i64, none) attributes {argNames = ["inCtrl"], resNames = ["out0", "outCtrl"]} {
  %0:4 = fork [4] %arg0 : none
  %1 = constant %0#2 {value = 1 : i64} : i64
  %2 = constant %0#1 {value = 42 : i64} : i64
  %3 = constant %0#0 {value = 1 : i64} : i64
  %4 = buffer [1] seq %13#0 {initValues = [0]} : i1
  %5:4 = fork [4] %4 : i1
  %6 = mux %5#3 [%0#3, %trueResult_2] : i1, none
  %7 = mux %5#2 [%2, %trueResult] : i1, i64
  %8:2 = fork [2] %7 : i64
  %9 = mux %5#1 [%3, %14#0] : i1, i64
  %10 = mux %5#0 [%1, %15] : i1, i64
  %11:2 = fork [2] %10 : i64
  %12 = arith.cmpi slt, %11#0, %8#0 : i64
  %13:5 = fork [5] %12 : i1
  %trueResult, %falseResult = cond_br %13#4, %8#1 : i64
  sink %falseResult : i64
  %trueResult_0, %falseResult_1 = cond_br %13#3, %9 : i64
  sink %falseResult_1 : i64
  %trueResult_2, %falseResult_3 = cond_br %13#2, %6 : none
  %trueResult_4, %falseResult_5 = cond_br %13#1, %11#1 : i64
  %14:2 = fork [2] %trueResult_0 : i64
  %15 = arith.addi %trueResult_4, %14#1 : i64
  return %falseResult_5, %falseResult_3 : i64, none
}
