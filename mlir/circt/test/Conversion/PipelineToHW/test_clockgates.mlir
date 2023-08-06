// RUN: circt-opt --lower-pipeline-to-hw %s | FileCheck %s
// RUN: circt-opt --lower-pipeline-to-hw="clock-gate-regs" %s | FileCheck %s --check-prefix=CGATE


// CHECK-LABEL:   hw.module @testSingle(
// CHECK-SAME:           %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out0: i32, out1: i1) {
// CHECK:           %[[VAL_5:.*]] = comb.sub %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_6:.*]] = hw.constant true
// CHECK:           %[[VAL_7:.*]] = hw.constant false
// CHECK:           %[[VAL_8:.*]] = seq.compreg sym @p0_stage0_reg0 %[[VAL_5]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_9:.*]] = seq.compreg sym @p0_stage0_reg1 %[[VAL_0]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_10:.*]] = hw.constant false
// CHECK:           %[[VAL_11:.*]] = seq.compreg sym @p0_stage0_valid %[[VAL_2]], %[[VAL_3]], %[[VAL_4]], %[[VAL_10]]  : i1
// CHECK:           %[[VAL_12:.*]] = comb.add %[[VAL_8]], %[[VAL_9]] : i32
// CHECK:           hw.output %[[VAL_12]], %[[VAL_11]] : i32, i1
// CHECK:         }

// CGATE-LABEL:   hw.module @testSingle(
// CGATE-SAME:           %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out0: i32, out1: i1) {
// CGATE:           %[[VAL_5:.*]] = comb.sub %[[VAL_0]], %[[VAL_1]] : i32
// CGATE:           %[[VAL_6:.*]] = hw.constant true
// CGATE:           %[[VAL_7:.*]] = hw.constant false
// CGATE:           %[[VAL_8:.*]] = seq.clock_gate %[[VAL_3]], %[[VAL_2]]
// CGATE:           %[[VAL_9:.*]] = seq.clock_gate %[[VAL_8]], %[[VAL_6]]
// CGATE:           %[[VAL_10:.*]] = seq.clock_gate %[[VAL_9]], %[[VAL_7]]
// CGATE:           %[[VAL_11:.*]] = seq.compreg sym @p0_stage0_reg0 %[[VAL_5]], %[[VAL_10]] : i32
// CGATE:           %[[VAL_12:.*]] = seq.compreg sym @p0_stage0_reg1 %[[VAL_0]], %[[VAL_8]] : i32
// CGATE:           %[[VAL_13:.*]] = hw.constant false
// CGATE:           %[[VAL_14:.*]] = seq.compreg sym @p0_stage0_valid %[[VAL_2]], %[[VAL_3]], %[[VAL_4]], %[[VAL_13]]  : i1
// CGATE:           %[[VAL_15:.*]] = comb.add %[[VAL_11]], %[[VAL_12]] : i32
// CGATE:           hw.output %[[VAL_15]], %[[VAL_14]] : i32, i1
// CGATE:         }

hw.module @testSingle(%arg0: i32, %arg1: i32, %go: i1, %clk: i1, %rst: i1) -> (out0: i32, out1: i1) {
  %0:2 = pipeline.scheduled(%a0 : i32 = %arg0, %a1 : i32 = %arg1) clock(%c = %clk) reset(%r = %rst) go(%g = %go) -> (out: i32){
    %1 = comb.sub %a0, %a1 : i32
    %true = hw.constant true
    %false = hw.constant false
    pipeline.stage ^bb1 regs(%1 : i32 gated by [%true, %false], %a0 : i32)
  ^bb1(%6: i32, %7: i32, %s1_valid : i1):  // pred: ^bb1
    %8 = comb.add %6, %7 : i32
    pipeline.return %8 : i32
  }
  hw.output %0#0, %0#1 : i32, i1
}
