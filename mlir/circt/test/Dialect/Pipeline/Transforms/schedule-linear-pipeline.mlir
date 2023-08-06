// RUN: circt-opt --pass-pipeline='builtin.module(any(pipeline-schedule-linear))' %s | FileCheck %s

// CHECK-LABEL:   hw.module @pipeline(
// CHECK-SAME:          %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out: i32) {
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = pipeline.scheduled(%[[VAL_7:.*]] : i32 = %[[VAL_0]], %[[VAL_8:.*]] : i32 = %[[VAL_1]]) clock(%[[VAL_9:.*]] = %[[VAL_3]]) reset(%[[VAL_10:.*]] = %[[VAL_4]]) go(%[[VAL_11:.*]] = %[[VAL_2]]) -> (out : i32) {
// CHECK:             %[[VAL_12:.*]] = comb.add %[[VAL_7]], %[[VAL_8]] {ssp.operator_type = @add1} : i32
// CHECK:             %[[VAL_13:.*]] = comb.add %[[VAL_8]], %[[VAL_7]] {ssp.operator_type = @add1} : i32
// CHECK:             pipeline.stage ^bb1
// CHECK:           ^bb1(%[[VAL_14:.*]]: i1):
// CHECK:             pipeline.stage ^bb2
// CHECK:           ^bb2(%[[VAL_15:.*]]: i1):
// CHECK:             %[[VAL_16:.*]] = comb.mul %[[VAL_7]], %[[VAL_12]] {ssp.operator_type = @mul2} : i32
// CHECK:             pipeline.stage ^bb3
// CHECK:           ^bb3(%[[VAL_17:.*]]: i1):
// CHECK:             pipeline.stage ^bb4
// CHECK:           ^bb4(%[[VAL_18:.*]]: i1):
// CHECK:             pipeline.stage ^bb5
// CHECK:           ^bb5(%[[VAL_19:.*]]: i1):
// CHECK:             %[[VAL_20:.*]] = comb.add %[[VAL_16]], %[[VAL_13]] {ssp.operator_type = @add1} : i32
// CHECK:             pipeline.stage ^bb6
// CHECK:           ^bb6(%[[VAL_21:.*]]: i1):
// CHECK:             pipeline.stage ^bb7
// CHECK:           ^bb7(%[[VAL_22:.*]]: i1):
// CHECK:             pipeline.return %[[VAL_20]] : i32
// CHECK:           }
// CHECK:           hw.output %[[VAL_23:.*]] : i32
// CHECK:         }

module {
  ssp.library @lib {
    operator_type @add1 [latency<2>]
    operator_type @mul2 [latency<3>]
  }

  hw.module @pipeline(%arg0 : i32, %arg1 : i32, %go : i1, %clk : i1, %rst : i1) -> (out: i32) {
    %0:2 = pipeline.unscheduled(%a0 : i32 = %arg0, %a1 : i32 = %arg1) clock(%c = %clk) reset(%r = %rst) go(%g = %go)
      {operator_lib = @lib} -> (out: i32) {
      %0 = comb.add %a0, %a1 {ssp.operator_type = @add1} : i32
      %1 = comb.mul %a0, %0 {ssp.operator_type = @mul2} : i32
      %2 = comb.add %a1, %a0 {ssp.operator_type = @add1} : i32
      %3 = comb.add %1, %2 {ssp.operator_type = @add1} : i32
      pipeline.return %3 : i32
    }
    hw.output %0#0 : i32
  }
}
