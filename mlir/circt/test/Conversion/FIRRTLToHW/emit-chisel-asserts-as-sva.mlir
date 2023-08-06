// RUN: circt-opt -lower-firrtl-to-hw=emit-chisel-asserts-as-sva %s | FileCheck %s

firrtl.circuit "ifElseFatalToSVA" {
  // CHECK-LABEL: hw.module @ifElseFatalToSVA
  firrtl.module @ifElseFatalToSVA(
    in %clock: !firrtl.clock,
    in %cond: !firrtl.uint<1>,
    in %enable: !firrtl.uint<1>
  ) {
    firrtl.assert %clock, %cond, %enable, "assert0" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {isConcurrent = true, format = "ifElseFatal"}
    // CHECK-NEXT: [[TRUE:%.+]] = hw.constant true
    // CHECK-NEXT: [[TMP1:%.+]] = comb.xor bin %enable, [[TRUE]]
    // CHECK-NEXT: [[TMP2:%.+]] = comb.or bin [[TMP1]], %cond
    // CHECK-NEXT: sv.assert.concurrent posedge %clock, [[TMP2]] message "assert0"
    // CHECK-NEXT: sv.ifdef "USE_PROPERTY_AS_CONSTRAINT" {
    // CHECK-NEXT:   sv.assume.concurrent posedge %clock, [[TMP2]]
    // CHECK-NEXT: }
  }

  // Test that an immediate assertion is always converted to a concurrent
  // assertion if the "emit-chisel-asserts-as-sva" option is enabled.
  //
  // CHECK-LABEL: hw.module @immediateToConcurrent
  firrtl.module @immediateToConcurrent(
    in %clock: !firrtl.clock,
    in %cond: !firrtl.uint<1>,
    in %enable: !firrtl.uint<1>
  ) {
    firrtl.assert %clock, %cond, %enable, "assert1" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: sv.assert.concurrent
  }
}
