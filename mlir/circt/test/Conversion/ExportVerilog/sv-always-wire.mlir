// RUN: circt-opt %s -prettify-verilog --export-verilog --verify-diagnostics -o %t | FileCheck %s --strict-whitespace
// RUN: circt-opt %s -test-apply-lowering-options='options=exprInEventControl' -prettify-verilog -export-verilog | FileCheck %s --check-prefix=INLINE

// CHECK-LABEL: module AlwaysSpill(
hw.module @AlwaysSpill(%port: i1) {
  %false = hw.constant false
  %true = hw.constant true
  %awire = sv.wire : !hw.inout<i1>

  // CHECK: wire [[TMP1:.+]] = 1'h0;
  // CHECK: wire [[TMP2:.+]] = 1'h1;
  // CHECK: wire{{ *}}awire;
  %awire2 = sv.read_inout %awire : !hw.inout<i1>

  // Existing simple names should not cause additional spill.
  // CHECK: always @(posedge port)
  sv.always posedge %port {}
  // CHECK: always_ff @(posedge port)
  sv.alwaysff(posedge %port) {}
  // CHECK: always @(posedge awire)
  sv.always posedge %awire2 {}
  // CHECK: always_ff @(posedge awire)
  sv.alwaysff(posedge %awire2) {}

  // Constant values should cause a spill.
  // CHECK: always @(posedge [[TMP1]])
  // INLINE: always @(posedge 1'h0)
  sv.always posedge %false {}
  // CHECK: always_ff @(posedge [[TMP2]])
  // INLINE: always_ff @(posedge 1'h1)
  sv.alwaysff(posedge %true) {}
}

// CHECK-LABEL: module Foo
// INLINE-LABEL: module Foo
hw.module @Foo(%reset0: i1, %reset1: i1) -> () {
  %0 = comb.or %reset0, %reset1 : i1
  // CHECK:      wire [[TMP0:.*]] = reset0 | reset1;
  // CHECK-NEXT: always @(posedge [[TMP0]])
  // CHECK-NEXT:   if ([[TMP0]])
  sv.always posedge %0 {
    sv.if %0 {
    }
  }
  %true = hw.constant true
  %1 = comb.xor %reset0, %true : i1
  // CHECK:      wire [[TMP1:.*]] = ~reset0;
  // CHECK-NEXT: always @(posedge [[TMP1]])
  // CHECK-NEXT:   if ([[TMP1]])
  // INLINE:     always @(posedge ~reset0)
  // INLINE-NEXT:   if (~reset0)
  sv.always posedge %1 {
    sv.if %1 {
    }
  }
}
