// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-expand-whens)))' --mlir-print-local-scope  --mlir-print-debuginfo %s | FileCheck %s

firrtl.circuit "Basic"  {
// CHECK-LABEL: @Basic
firrtl.module @Basic(in %p: !firrtl.uint<1>, in %v0: !firrtl.uint<8>, in %v1: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {
  firrtl.when %p : !firrtl.uint<1> {
    firrtl.connect %out, %v0 : !firrtl.uint<8>, !firrtl.uint<8> loc("then")
  } else {
    firrtl.connect %out, %v1 : !firrtl.uint<8>, !firrtl.uint<8> loc("else")
  } loc("when")
  // CHECK: [[MUX:%.*]] = firrtl.mux(%p, %v0, %v1)
  // CHECK-SAME: loc(fused["when", "then", "else"])

  // CHECK: firrtl.connect %out, [[MUX]]
  // CHECK-SAME: loc("when")
}

// CHECK-LABEL: @Default
firrtl.module @Default(in %p: !firrtl.uint<1>, in %v0: !firrtl.uint<8>, in %v1: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {
  firrtl.connect %out, %v1 : !firrtl.uint<8>, !firrtl.uint<8> loc("else")
  firrtl.when %p : !firrtl.uint<1> {
    firrtl.connect %out, %v0 : !firrtl.uint<8>, !firrtl.uint<8> loc("then")
  } loc("when")
  // CHECK: [[MUX:%.*]] = firrtl.mux(%p, %v0, %v1)
  // CHECK-SAME: loc(fused["when", "then", "else"])

  // CHECK: firrtl.connect %out, [[MUX]]
  // CHECK-SAME: loc("when")
}

// CHECK-LABEL: @Nested
firrtl.module @Nested(in %p: !firrtl.uint<1>, in %v0: !firrtl.uint<8>, in %v1: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {
  firrtl.connect %out, %v1 : !firrtl.uint<8>, !firrtl.uint<8> loc("else")
  firrtl.when %p : !firrtl.uint<1> {
    firrtl.when %p : !firrtl.uint<1> {
      firrtl.connect %out, %v0 : !firrtl.uint<8>, !firrtl.uint<8> loc("then")
    } else {
    } loc("inside")
  } loc("outside")
  
  // CHECK: [[INSIDE:%.*]] = firrtl.mux(%p, %v0, %v1)
  // CHECK-SAME: loc(fused["inside", "then", "else"])
  
  // CHECK: [[OUTSIDE:%.*]] = firrtl.mux(%p, [[INSIDE]], %v1)
  // CHECK-SAME: loc(fused["outside", "inside", "else"])
  
  // CHECK: firrtl.connect %out, [[OUTSIDE]]
  // CHECK-SAME: loc("outside")
}
}
