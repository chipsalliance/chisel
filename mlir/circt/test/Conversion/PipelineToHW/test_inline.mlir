// RUN: circt-opt --lower-pipeline-to-hw %s | FileCheck %s

// CHECK-LABEL:   hw.module @testBasic(
// CHECK-SAME:           %[[VAL_0:.*]]: i1, %[[VAL_1:.*]]: i1, %[[VAL_2:.*]]: i1) -> (out: i1) {
// CHECK:           hw.output %[[VAL_0]] : i1
// CHECK:         }
hw.module @testBasic(%arg0: i1, %clk: i1, %rst: i1) -> (out: i1) {
  %0:2 = pipeline.scheduled(%a0 : i1 = %arg0) clock(%c = %clk) reset(%r = %rst) go(%g = %arg0) -> (out : i1) {
    pipeline.return %a0 : i1
  }
  hw.output %0#0 : i1
}

// CHECK-LABEL:   hw.module @testLatency1(
// CHECK-SAME:            %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out: i32, done: i1) {
// CHECK:           %[[VAL_5:.*]] = comb.add %[[VAL_0]], %[[VAL_0]] : i32
// CHECK:           %[[VAL_6:.*]] = hw.constant false
// CHECK:           %[[VAL_7:.*]] = seq.compreg sym @p0_stage0_valid %[[VAL_2]], %[[VAL_3]], %[[VAL_4]], %[[VAL_6]]  : i1
// CHECK:           %[[VAL_8:.*]] = hw.constant false
// CHECK:           %[[VAL_9:.*]] = seq.compreg sym @p0_stage1_valid %[[VAL_7]], %[[VAL_3]], %[[VAL_4]], %[[VAL_8]]  : i1
// CHECK:           %[[VAL_10:.*]] = seq.compreg sym @p0_stage2_reg0 %[[VAL_5]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_11:.*]] = hw.constant false
// CHECK:           %[[VAL_12:.*]] = seq.compreg sym @p0_stage2_valid %[[VAL_9]], %[[VAL_3]], %[[VAL_4]], %[[VAL_11]]  : i1
// CHECK:           %[[VAL_13:.*]] = seq.compreg sym @p0_stage3_reg0 %[[VAL_10]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_14:.*]] = hw.constant false
// CHECK:           %[[VAL_15:.*]] = seq.compreg sym @p0_stage3_valid %[[VAL_12]], %[[VAL_3]], %[[VAL_4]], %[[VAL_14]]  : i1
// CHECK:           hw.output %[[VAL_13]], %[[VAL_15]] : i32, i1
// CHECK:         }
hw.module @testLatency1(%arg0: i32, %arg1: i32, %go: i1, %clk: i1, %rst: i1) -> (out: i32, done: i1) {
  %out, %done = pipeline.scheduled(%a0 : i32 = %arg0) clock(%c = %clk) reset(%r = %rst) go(%g = %go) -> (out: i32) {
    %1 = pipeline.latency 2 -> (i32) {
      %6 = comb.add %a0, %a0 : i32
      pipeline.latency.return %6 : i32
    }
    pipeline.stage ^bb1 pass(%1 : i32)
  ^bb1(%2: i32, %s1_valid: i1):  // pred: ^bb0
    pipeline.stage ^bb2 pass(%2 : i32)
  ^bb2(%3: i32, %s2_valid: i1):  // pred: ^bb1
    pipeline.stage ^bb3 regs(%3 : i32)
  ^bb3(%4: i32, %s3_valid: i1):  // pred: ^bb2
    pipeline.stage ^bb4 regs(%4 : i32)
  ^bb4(%5: i32, %s4_valid: i1):  // pred: ^bb3
    pipeline.return %5 : i32
  }
  hw.output %out, %done : i32, i1
}

// CHECK-LABEL:   hw.module @testSingle(
// CHECK-SAME:          %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out0: i32, out1: i1) {
// CHECK:           %[[VAL_5:.*]] = comb.sub %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_6:.*]] = seq.compreg sym @p0_stage0_reg0 %[[VAL_5]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_7:.*]] = seq.compreg sym @p0_stage0_reg1 %[[VAL_0]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_8:.*]] = hw.constant false
// CHECK:           %[[VAL_9:.*]] = seq.compreg sym @p0_stage0_valid %[[VAL_2]], %[[VAL_3]], %[[VAL_4]], %[[VAL_8]]  : i1
// CHECK:           %[[VAL_10:.*]] = comb.add %[[VAL_6]], %[[VAL_7]] : i32
// CHECK:           hw.output %[[VAL_10]], %[[VAL_9]] : i32, i1
// CHECK:         }
hw.module @testSingle(%arg0: i32, %arg1: i32, %go: i1, %clk: i1, %rst: i1) -> (out0: i32, out1: i1) {
  %0:2 = pipeline.scheduled(%a0 : i32 = %arg0, %a1 : i32 = %arg1) clock(%c = %clk) reset(%r = %rst) go(%g = %go) -> (out: i32){
    %1 = comb.sub %a0,%a1 : i32
    pipeline.stage ^bb1 regs(%1 : i32, %a0 : i32)
  ^bb1(%6: i32, %7: i32, %s1_valid : i1):  // pred: ^bb1
    %8 = comb.add %6, %7 : i32
    pipeline.return %8 : i32
  }
  hw.output %0#0, %0#1 : i32, i1
}

// CHECK-LABEL:   hw.module @testMultiple(
// CHECK-SAME:          %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out0: i32, out1: i1) {
// CHECK:           %[[VAL_5:.*]] = comb.sub %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_6:.*]] = seq.compreg sym @p0_stage0_reg0 %[[VAL_5]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_7:.*]] = seq.compreg sym @p0_stage0_reg1 %[[VAL_0]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_8:.*]] = hw.constant false
// CHECK:           %[[VAL_9:.*]] = seq.compreg sym @p0_stage0_valid %[[VAL_2]], %[[VAL_3]], %[[VAL_4]], %[[VAL_8]]  : i1
// CHECK:           %[[VAL_10:.*]] = comb.add %[[VAL_6]], %[[VAL_7]] : i32
// CHECK:           %[[VAL_11:.*]] = seq.compreg sym @p0_stage1_reg0 %[[VAL_10]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_12:.*]] = seq.compreg sym @p0_stage1_reg1 %[[VAL_6]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_13:.*]] = hw.constant false
// CHECK:           %[[VAL_14:.*]] = seq.compreg sym @p0_stage1_valid %[[VAL_9]], %[[VAL_3]], %[[VAL_4]], %[[VAL_13]]  : i1
// CHECK:           %[[VAL_15:.*]] = comb.mul %[[VAL_11]], %[[VAL_12]] : i32
// CHECK:           %[[VAL_16:.*]] = comb.sub %[[VAL_15]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_17:.*]] = seq.compreg sym @p1_stage0_reg0 %[[VAL_16]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_18:.*]] = seq.compreg sym @p1_stage0_reg1 %[[VAL_15]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_19:.*]] = hw.constant false
// CHECK:           %[[VAL_20:.*]] = seq.compreg sym @p1_stage0_valid %[[VAL_2]], %[[VAL_3]], %[[VAL_4]], %[[VAL_19]]  : i1
// CHECK:           %[[VAL_21:.*]] = comb.add %[[VAL_17]], %[[VAL_18]] : i32
// CHECK:           %[[VAL_22:.*]] = seq.compreg sym @p1_stage1_reg0 %[[VAL_21]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_23:.*]] = seq.compreg sym @p1_stage1_reg1 %[[VAL_17]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_24:.*]] = hw.constant false
// CHECK:           %[[VAL_25:.*]] = seq.compreg sym @p1_stage1_valid %[[VAL_20]], %[[VAL_3]], %[[VAL_4]], %[[VAL_24]]  : i1
// CHECK:           %[[VAL_26:.*]] = comb.mul %[[VAL_22]], %[[VAL_23]] : i32
// CHECK:           hw.output %[[VAL_15]], %[[VAL_14]] : i32, i1
// CHECK:         }
hw.module @testMultiple(%arg0: i32, %arg1: i32, %go: i1, %clk: i1, %rst: i1) -> (out0: i32, out1: i1) {
  %0:2 = pipeline.scheduled(%a0 : i32 = %arg0, %a1 : i32 = %arg1) clock(%c = %clk) reset(%r = %rst) go(%g = %go) -> (out: i32){
    %1 = comb.sub %a0,%a1 : i32
    pipeline.stage ^bb1 regs(%1 : i32, %a0 : i32)
  ^bb1(%2: i32, %3: i32, %s1_valid: i1):  // pred: ^bb0
    %5 = comb.add %2, %3 : i32
    pipeline.stage ^bb2 regs(%5 : i32, %2 : i32)
  ^bb2(%6: i32, %7: i32, %s2_valid: i1):  // pred: ^bb1
    %8 = comb.mul %6, %7 : i32
    pipeline.return %8 : i32
  }

  %1:2 = pipeline.scheduled(%a0 : i32 = %0#0, %a1 : i32 = %arg1) clock(%c = %clk) reset(%r = %rst) go(%g = %go) -> (out: i32){
    %1 = comb.sub %a0,%a1 : i32
    pipeline.stage ^bb1 regs(%1 : i32, %a0 : i32)
  ^bb1(%2: i32, %3: i32, %s1_valid: i1):  // pred: ^bb0
    %5 = comb.add %2, %3 : i32
    pipeline.stage ^bb2 regs(%5 : i32, %2 : i32)
  ^bb2(%6: i32, %7: i32, %s2_valid: i1):  // pred: ^bb1
    %8 = comb.mul %6, %7 : i32
    pipeline.return %8 : i32
  }

  hw.output %0#0, %0#1 : i32, i1
}

// CHECK-LABEL:   hw.module @testSingleWithExt(
// CHECK-SAME:                                 %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out0: i32, out1: i32) {
// CHECK:           %[[VAL_5:.*]] = hw.constant true
// CHECK:           %[[VAL_6:.*]] = comb.sub %[[VAL_0]], %[[VAL_0]] : i32
// CHECK:           %[[VAL_7:.*]] = seq.compreg sym @p0_stage0_reg0 %[[VAL_6]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_8:.*]] = hw.constant false
// CHECK:           %[[VAL_9:.*]] = seq.compreg sym @p0_stage0_valid %[[VAL_2]], %[[VAL_3]], %[[VAL_4]], %[[VAL_8]]  : i1
// CHECK:           %[[VAL_10:.*]] = comb.add %[[VAL_7]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_11:.*]] = seq.compreg sym @p0_stage1_reg0 %[[VAL_10]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_12:.*]] = hw.constant false
// CHECK:           %[[VAL_13:.*]] = seq.compreg sym @p0_stage1_valid %[[VAL_9]], %[[VAL_3]], %[[VAL_4]], %[[VAL_12]]  : i1
// CHECK:           hw.output %[[VAL_11]], %[[VAL_1]] : i32, i32
// CHECK:         }
hw.module @testSingleWithExt(%arg0: i32, %ext1: i32, %go : i1, %clk: i1, %rst: i1) -> (out0: i32, out1: i32) {
  %0:3 = pipeline.scheduled(%a0 : i32 = %arg0, %a1 : i32 = %arg0) ext (%e0 : i32 = %ext1) clock(%c = %clk) reset(%r = %rst) go(%g = %go) -> (out0: i32, out1: i32) {
    %true = hw.constant true
    %1 = comb.sub %a0, %a0 : i32
    pipeline.stage ^bb1 regs(%1 : i32)

  ^bb1(%6: i32, %s1_valid: i1):
    // Use the external value inside a stage
    %8 = comb.add %6, %e0 : i32
    pipeline.stage ^bb2 regs(%8 : i32)
  
  ^bb2(%9 : i32, %s2_valid: i1):
  // Use the external value in the exit stage.
    pipeline.return %9, %e0 : i32, i32
  }
  hw.output %0#0, %0#1 : i32, i32
}

// CHECK-LABEL:   hw.module @testControlUsage(
// CHECK-SAME:              %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i1, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1) -> (out0: i32) {
// CHECK:           %[[VAL_4:.*]] = hw.constant 0 : i32
// CHECK:           %[[VAL_5:.*]] = sv.wire : !hw.inout<i32>
// CHECK:           %[[VAL_6:.*]] = sv.read_inout %[[VAL_5]] : !hw.inout<i32>
// CHECK:           %[[VAL_7:.*]] = comb.add %[[VAL_6]], %[[VAL_0]] : i32
// CHECK:           %[[VAL_8:.*]] = seq.compreg.ce %[[VAL_7]], %[[VAL_2]], %[[VAL_1]], %[[VAL_3]], %[[VAL_4]]  : i32
// CHECK:           sv.assign %[[VAL_5]], %[[VAL_8]] : i32
// CHECK:           %[[VAL_9:.*]] = seq.compreg sym @p0_stage0_reg0 %[[VAL_8]], %[[VAL_2]] : i32
// CHECK:           %[[VAL_10:.*]] = hw.constant false
// CHECK:           %[[VAL_11:.*]] = seq.compreg sym @p0_stage0_valid %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_10]]  : i1
// CHECK:           %[[VAL_12:.*]] = sv.wire : !hw.inout<i32>
// CHECK:           %[[VAL_13:.*]] = sv.read_inout %[[VAL_12]] : !hw.inout<i32>
// CHECK:           %[[VAL_14:.*]] = comb.add %[[VAL_13]], %[[VAL_9]] : i32
// CHECK:           %[[VAL_15:.*]] = seq.compreg.ce %[[VAL_14]], %[[VAL_2]], %[[VAL_11]], %[[VAL_3]], %[[VAL_4]]  : i32
// CHECK:           sv.assign %[[VAL_12]], %[[VAL_15]] : i32
// CHECK:           %[[VAL_16:.*]] = seq.compreg sym @p0_stage1_reg0 %[[VAL_15]], %[[VAL_2]] : i32
// CHECK:           %[[VAL_17:.*]] = hw.constant false
// CHECK:           %[[VAL_18:.*]] = seq.compreg sym @p0_stage1_valid %[[VAL_11]], %[[VAL_2]], %[[VAL_3]], %[[VAL_17]]  : i1
// CHECK:           %[[VAL_19:.*]] = sv.wire : !hw.inout<i32>
// CHECK:           %[[VAL_20:.*]] = sv.read_inout %[[VAL_19]] : !hw.inout<i32>
// CHECK:           %[[VAL_21:.*]] = comb.add %[[VAL_20]], %[[VAL_16]] : i32
// CHECK:           %[[VAL_22:.*]] = seq.compreg.ce %[[VAL_21]], %[[VAL_2]], %[[VAL_18]], %[[VAL_3]], %[[VAL_4]]  : i32
// CHECK:           sv.assign %[[VAL_19]], %[[VAL_22]] : i32
// CHECK:           hw.output %[[VAL_22]] : i32
// CHECK:         }
hw.module @testControlUsage(%arg0: i32, %go : i1, %clk: i1, %rst: i1) -> (out0: i32) {
  %0:2 = pipeline.scheduled(%a0 : i32 = %arg0) clock(%c = %clk) reset(%r = %rst) go(%g = %go) -> (out: i32) {
    %zero = hw.constant 0 : i32
    %reg_out_wire = sv.wire : !hw.inout<i32>
    %reg_out = sv.read_inout %reg_out_wire : !hw.inout<i32>
    %add0 = comb.add %reg_out, %a0 : i32
    %out = seq.compreg.ce %add0, %c, %g, %r, %zero : i32
    sv.assign %reg_out_wire, %out : i32
    pipeline.stage ^bb1 regs(%out : i32)

  ^bb1(%6: i32, %s1_valid: i1):
    %reg1_out_wire = sv.wire : !hw.inout<i32>
    %reg1_out = sv.read_inout %reg1_out_wire : !hw.inout<i32>
    %add1 = comb.add %reg1_out, %6 : i32
    %out1 = seq.compreg.ce %add1, %c, %s1_valid, %r, %zero : i32
    sv.assign %reg1_out_wire, %out1 : i32

    pipeline.stage ^bb2 regs(%out1 : i32)
  
  ^bb2(%9 : i32, %s2_valid: i1):
    %reg2_out_wire = sv.wire : !hw.inout<i32>
    %reg2_out = sv.read_inout %reg2_out_wire : !hw.inout<i32>
    %add2 = comb.add %reg2_out, %9 : i32
    %out2 = seq.compreg.ce %add2, %c, %s2_valid, %r, %zero : i32
    sv.assign %reg2_out_wire, %out2 : i32
    pipeline.return %out2  : i32
  }
  hw.output %0#0 : i32
}

// -----

// CHECK-LABEL:   hw.module @testWithStall(
// CHECK-SAME:            %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i1, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out0: i32, out1: i1) {
// CHECK:           %[[VAL_5:.*]] = hw.constant true
// CHECK:           %[[VAL_6:.*]] = comb.xor %[[VAL_2]], %[[VAL_5]] : i1
// CHECK:           %[[VAL_7:.*]] = comb.and %[[VAL_1]], %[[VAL_6]] : i1
// CHECK:           %[[VAL_8:.*]] = seq.compreg.ce sym @p0_stage0_reg0 %[[VAL_0]], %[[VAL_3]], %[[VAL_7]] : i32
// CHECK:           %[[VAL_9:.*]] = hw.constant false
// CHECK:           %[[VAL_10:.*]] = seq.compreg.ce sym @p0_stage0_valid %[[VAL_1]], %[[VAL_3]], %[[VAL_6]], %[[VAL_4]], %[[VAL_9]]  : i1
// CHECK:           hw.output %[[VAL_8]], %[[VAL_10]] : i32, i1
// CHECK:         }
hw.module @testWithStall(%arg0: i32, %go: i1, %stall : i1, %clk: i1, %rst: i1) -> (out0: i32, out1: i1) {
  %0:2 = pipeline.scheduled(%a0 : i32 = %arg0) stall(%s = %stall) clock(%c = %clk) reset(%r = %rst) go(%g = %go) -> (out: i32) {
    pipeline.stage ^bb1 regs(%a0 : i32)
  ^bb1(%1: i32, %s1_valid : i1):  // pred: ^bb1
    pipeline.return %1 : i32
  }
  hw.output %0#0, %0#1 : i32, i1
}
