// RUN: circt-opt -split-input-file -verify-diagnostics %s


hw.module @res_argn(%arg0 : i32, %arg1 : i32, %go : i1, %clk : i1, %rst : i1) -> (out: i32) {
  %0:2 = pipeline.unscheduled(%a0 : i32 = %arg0, %a1 : i32 = %arg1) clock(%c = %clk) reset(%r = %rst) go(%g = %go) -> (out: i32){
    // expected-error @+1 {{'pipeline.return' op expected 1 return values, got 0.}}
    pipeline.return
  }
  hw.output %0#0 : i32
}

// -----

hw.module @res_argtype(%arg0 : i32, %arg1 : i32, %go : i1, %clk : i1, %rst : i1) -> (out: i31) {
  %0:2 = pipeline.unscheduled(%a0 : i32 = %arg0, %a1 : i32 = %arg1) clock(%c = %clk) reset(%r = %rst) go(%g = %go) -> (out: i31) {
    // expected-error @+1 {{'pipeline.return' op expected return value of type 'i31', got 'i32'.}}
    pipeline.return %a0 : i32
  }
  hw.output %0#0 : i31
}

// -----

hw.module @unterminated(%arg0 : i32, %arg1 : i32, %go : i1, %clk : i1, %rst : i1) -> (out: i32) {
  // expected-error @+1 {{'pipeline.scheduled' op all blocks must be terminated with a `pipeline.stage` or `pipeline.return` op.}}
  %0:2 = pipeline.scheduled(%a0 : i32 = %arg0, %a1 : i32 = %arg1) clock(%c = %clk) reset(%r = %rst) go(%g = %go) -> (out: i32){
    %0 = comb.add %a0, %a1 : i32

  ^bb1(%s1_valid : i1):
    pipeline.stage ^bb2 regs(%0 : i32)

  ^bb2(%s2_s0 : i32, %s2_valid : i1):
    pipeline.return %s2_s0 : i32
  }
  hw.output %0#0 : i32
}

// -----

hw.module @mixed_stages(%arg0 : i32, %arg1 : i32, %go : i1, %clk : i1, %rst : i1) -> (out: i32) {
  %0:2 = pipeline.scheduled(%a0 : i32 = %arg0, %a1 : i32 = %arg1) clock(%c = %clk) reset(%r = %rst) go(%g = %go) -> (out: i32){
    %0 = comb.add %a0, %a1 : i32
    pipeline.stage ^bb1

  ^bb1(%s1_valid : i1):
  // expected-error @+1 {{'pipeline.stage' op Pipeline is in register materialized mode - operand 0 is defined in a different stage, which is illegal.}}
    pipeline.stage ^bb2 regs(%0: i32)

  ^bb2(%s2_s0 : i32, %s2_valid : i1):
    pipeline.return %s2_s0 : i32
  }
  hw.output %0#0 : i32
}

// -----

hw.module @cycle_pipeline1(%arg0 : i32, %arg1 : i32, %go : i1, %clk : i1, %rst : i1) -> (out: i32) {
  // expected-error @+1 {{'pipeline.scheduled' op pipeline contains a cycle.}}
  %0:2 = pipeline.scheduled(%a0 : i32 = %arg0, %a1 : i32 = %arg1) clock(%c = %clk) reset(%r = %rst) go(%g = %go) -> (out: i32){
    %0 = comb.add %a0, %a1 : i32
    pipeline.stage ^bb1

  ^bb1(%s1_valid : i1):
    pipeline.stage ^bb2

  ^bb2(%s2_valid : i1):
    pipeline.stage ^bb1

  ^bb3(%s3_valid : i1):
    pipeline.return %0 : i32
  }
  hw.output %0#0 : i32
}

// -----

hw.module @cycle_pipeline2(%arg0 : i32, %arg1 : i32, %go : i1, %clk : i1, %rst : i1) -> (out: i32) {
  // expected-error @+1 {{'pipeline.scheduled' op pipeline contains a cycle.}}
  %0:2 = pipeline.scheduled(%a0 : i32 = %arg0, %a1 : i32 = %arg1) clock(%c = %clk) reset(%r = %rst) go(%g = %go) -> (out: i32){
    %0 = comb.add %a0, %a1 : i32
    pipeline.stage ^bb1

  ^bb1(%s1_valid : i1):
    pipeline.stage ^bb1

  ^bb3(%s3_valid : i1):
    pipeline.return %0 : i32
  }
  hw.output %0#0 : i32
}

// -----

hw.module @earlyAccess(%arg0: i32, %arg1: i32, %go: i1, %clk: i1, %rst: i1) -> (out: i32) {
  %0:2 = pipeline.scheduled(%a0 : i32 = %arg0) clock(%c = %clk) reset(%r = %rst) go(%g = %go) -> (out: i32) {
    // expected-error @+1 {{'pipeline.latency' op result 0 is used before it is available.}}
    %1 = pipeline.latency 2 -> (i32) {
      %6 = comb.add %a0, %a0 : i32
      pipeline.latency.return %6 : i32
    }
    pipeline.stage ^bb1
  ^bb1(%s1_valid : i1):
    // expected-note@+1 {{use was operand 0. The result is available 1 stages later than this use.}}
    pipeline.return %1 : i32
  }
  hw.output %0#0 : i32
}

// -----

// Test which verifies that the values referenced within the body of a
// latency operation also adhere to the latency constraints.
hw.module @earlyAccess2(%arg0: i32, %arg1: i32, %go: i1, %clk: i1, %rst: i1) -> (out: i32) {
  %0:2 = pipeline.scheduled(%a0 : i32 = %arg0) clock(%c = %clk) reset(%r = %rst) go(%g = %go) -> (out: i32) {
    // expected-error @+1 {{'pipeline.latency' op result 0 is used before it is available.}}
    %1 = pipeline.latency 2 -> (i32) {
      %res = comb.add %a0, %a0 : i32
      pipeline.latency.return %res : i32
    }
    pipeline.stage ^bb1

  ^bb1(%s1_valid : i1):
    %2 = pipeline.latency 2 -> (i32) {
      %c1_i32 = hw.constant 1 : i32
      // expected-note@+1 {{use was operand 0. The result is available 1 stages later than this use.}}
      %res2 = comb.add %1, %c1_i32 : i32
      pipeline.latency.return %res2 : i32
    }
    pipeline.stage ^bb2

  ^bb2(%s2_valid : i1):
    pipeline.stage ^bb3

  ^bb3(%s3_valid : i1):
    pipeline.return %2 : i32
  }
  hw.output %0#0 : i32
}

// -----


hw.module @registeredPass(%arg0: i32, %arg1: i32, %go: i1, %clk: i1, %rst: i1) -> (out: i32) {
  %0:2 = pipeline.scheduled(%a0 : i32 = %arg0) clock(%c = %clk) reset(%r = %rst) go(%g = %go) -> (out: i32) {
    // expected-error @+1 {{'pipeline.latency' op result 0 is used before it is available.}}
    %1 = pipeline.latency 2 -> (i32) {
      %6 = comb.add %a0, %a0 : i32
      pipeline.latency.return %6 : i32
    }
    // expected-note@+1 {{use was operand 0. The result is available 2 stages later than this use.}}
    pipeline.stage ^bb1 regs(%1 : i32)
  ^bb1(%v : i32, %s1_valid : i1):
    pipeline.return %v : i32
  }
  hw.output %0#0 : i32
}

// -----

hw.module @missing_valid_entry3(%arg : i32, %go : i1, %clk : i1, %rst : i1) -> () {
  // expected-error @+1 {{'pipeline.scheduled' op block 1 must have an i1 argument as the last block argument (stage valid signal).}}
  %done = pipeline.scheduled(%a0 : i32 = %arg) clock(%c = %clk) reset(%r = %rst) go(%g = %go) -> () {
     pipeline.stage ^bb1
   ^bb1:
      pipeline.return
  }
  hw.output
}

// -----

hw.module @invalid_clock_gate(%arg : i32, %go : i1, %clk : i1, %rst : i1) -> () {
  %done = pipeline.scheduled(%a0 : i32 = %arg) clock(%c = %clk) reset(%r = %rst) go(%g = %go) -> () {
     // expected-note@+1 {{prior use here}}
     %c0_i2 = hw.constant 0 : i2
     // expected-error @+1 {{use of value '%c0_i2' expects different type than prior uses: 'i1' vs 'i2'}}
     pipeline.stage ^bb1 regs(%a0 : i32 gated by [%c0_i2])
   ^bb1(%0 : i32, %s1_valid : i1):
      pipeline.return
  }
  hw.output
}
