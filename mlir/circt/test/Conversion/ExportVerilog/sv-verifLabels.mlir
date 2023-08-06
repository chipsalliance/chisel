// RUN: circt-opt --export-verilog %s | FileCheck %s --check-prefix=CHECK-OFF
// RUN: circt-opt --test-apply-lowering-options='options=verifLabels' --export-verilog %s | FileCheck %s --check-prefix=CHECK-ON

hw.module @foo(%clock: i1, %cond: i1) {
  sv.initial {
    // CHECK-OFF: assert(
    // CHECK-OFF: assume(
    // CHECK-OFF: cover(
    // CHECK-ON:  assert_0: assert(
    // CHECK-ON:  assume_0: assume(
    // CHECK-ON:  cover_0: cover(
    sv.assert %cond, immediate
    sv.assume %cond, immediate
    sv.cover %cond, immediate
  }
  // CHECK-OFF: assert property
  // CHECK-OFF: assume property
  // CHECK-OFF: cover property
  // CHECK-ON:  assert_1: assert property
  // CHECK-ON:  assume_1: assume property
  // CHECK-ON:  cover_1: cover property
  sv.assert.concurrent posedge %clock, %cond
  sv.assume.concurrent posedge %clock, %cond
  sv.cover.concurrent posedge %clock, %cond

  // Explicitly labeled ops should keep their label.
  sv.initial {
    // CHECK-OFF: imm_assert: assert(
    // CHECK-ON:  imm_assert: assert(
    // CHECK-OFF: imm_assume: assume(
    // CHECK-ON:  imm_assume: assume(
    // CHECK-OFF: imm_cover: cover(
    // CHECK-ON:  imm_cover: cover(
    sv.assert %cond, immediate label "imm_assert"
    sv.assume %cond, immediate label "imm_assume"
    sv.cover %cond, immediate label "imm_cover"
  }
  // CHECK-OFF: con_assert: assert property
  // CHECK-ON:  con_assert: assert property
  // CHECK-OFF: con_assume: assume property
  // CHECK-ON:  con_assume: assume property
  // CHECK-OFF: con_cover: cover property
  // CHECK-ON:  con_cover: cover property
  sv.assert.concurrent posedge %clock, %cond label "con_assert"
  sv.assume.concurrent posedge %clock, %cond label "con_assume"
  sv.cover.concurrent posedge %clock, %cond label "con_cover"

  // Explicitly labeled ops that conflict with implicit labels should force the
  // implicit labels to change, even if they appear earlier in the output.
  sv.initial {
    // CHECK-OFF: assert_0: assert(
    // CHECK-ON:  assert_0_0: assert(
    // CHECK-OFF: assume_2: assume(
    // CHECK-ON:  assume_2: assume(
    // CHECK-OFF: cover_4: cover(
    // CHECK-ON:  cover_4: cover(
    sv.assert %cond, immediate label "assert_0"
    sv.assume %cond, immediate label "assume_2"
    sv.cover %cond, immediate label "cover_4"
  }
  // CHECK-OFF: assert_6: assert property
  // CHECK-ON:  assert_6: assert property
  // CHECK-OFF: assume_8: assume property
  // CHECK-ON:  assume_8: assume property
  // CHECK-OFF: cover_10: cover property
  // CHECK-ON:  cover_10: cover property
  sv.assert.concurrent posedge %clock, %cond label "assert_6"
  sv.assume.concurrent posedge %clock, %cond label "assume_8"
  sv.cover.concurrent posedge %clock, %cond label "cover_10"
}
