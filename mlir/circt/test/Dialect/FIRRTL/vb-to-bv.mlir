// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-vb-to-bv))' %s | FileCheck %s

firrtl.circuit "Test" {
  firrtl.module @Test() {}
  //===--------------------------------------------------------------------===//
  // Port Type-Change Tests
  //===--------------------------------------------------------------------===//

  // CHECK:     @VG(in %port: !firrtl.vector<uint<8>, 4>)
  firrtl.module @VG(in %port: !firrtl.vector<uint<8>, 4>) {}

  // CHECK:      BG(in %port: !firrtl.bundle<a: uint<8>>)
  firrtl.module @BG(in %port: !firrtl.bundle<a: uint<8>>) {}

  // CHECK:     @VB(in %port: !firrtl.bundle<a: vector<uint<8>, 4>>)
  firrtl.module @VB(in %port: !firrtl.vector<bundle<a: uint<8>>, 4>) {}

  // CHECK:     @VB2(in %port: !firrtl.bundle<a: vector<uint<8>, 4>>)
  firrtl.module @VB2(in %port: !firrtl.vector<bundle<a: uint<8>>, 4>) {}

  // CHECK:     @VBB(in %port: !firrtl.bundle<nested: bundle<field: vector<uint<1>, 4>>>)
  firrtl.module @VBB(in %port: !firrtl.vector<bundle<nested: bundle<field: uint<1>>>, 4>) {}

  // CHECK:     @VVB(in %port: !firrtl.bundle<field: vector<vector<uint<1>, 6>, 4>>)
  firrtl.module @VVB(in %port: !firrtl.vector<vector<bundle<field: uint<1>>, 6>, 4>) {}

  // CHECK:     @VBVB(in %port: !firrtl.bundle<field_a: bundle<field_b: vector<vector<uint<1>, 4>, 8>>>)    
  firrtl.module @VBVB(in %port: !firrtl.vector<bundle<field_a: vector<bundle<field_b: uint<1>>, 4>>, 8>) {}

  //===--------------------------------------------------------------------===//
  // Aggregate Create/Constant Ops
  //===--------------------------------------------------------------------===//

  // CHECK-LABEL: @TestAggregateConstants
  firrtl.module @TestAggregateConstants() {
    // CHECK{LITERAL}: firrtl.aggregateconstant [1, 2, 3] : !firrtl.bundle<a: uint<8>, b: uint<5>, c: uint<4>>
    firrtl.aggregateconstant [1, 2, 3] : !firrtl.bundle<a: uint<8>, b: uint<5>, c: uint<4>>
    // CHECK{LITERAL}: firrtl.aggregateconstant [1, 2, 3] : !firrtl.vector<uint<8>, 3>
    firrtl.aggregateconstant [1, 2, 3] : !firrtl.vector<uint<8>, 3>
    // CHECK{LITERAL}: firrtl.aggregateconstant [[1, 3], [2, 4]] : !firrtl.bundle<a: vector<uint<8>, 2>, b: vector<uint<5>, 2>>
    firrtl.aggregateconstant [[1, 2], [3, 4]] : !firrtl.vector<bundle<a: uint<8>, b: uint<5>>, 2>
    // CHECK{LITERAL}: firrtl.aggregateconstant [[[1, 3], [2, 4]], [[5, 7], [6, 8]]] : !firrtl.bundle<a: bundle<c: vector<uint<8>, 2>, d: vector<uint<5>, 2>>, b: bundle<e: vector<uint<8>, 2>, f: vector<uint<5>, 2>>>
    firrtl.aggregateconstant [[[1, 2], [3, 4]], [[5, 6], [7, 8]]] : !firrtl.bundle<a: vector<bundle<c: uint<8>, d: uint<5>>, 2>, b: vector<bundle<e: uint<8>, f: uint<5>>, 2>>
    // CHECK{LITERAL}: firrtl.aggregateconstant [[[1, 3], [5, 7], [9, 11]], [[2, 4], [6, 8], [10, 12]]] : !firrtl.bundle<a: vector<vector<uint<8>, 2>, 3>, b: vector<vector<uint<8>, 2>, 3>>
    firrtl.aggregateconstant [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]] : !firrtl.vector<vector<bundle<a: uint<8>, b: uint<8>>, 2>, 3>
  }

  // CHECK-LABEL: @TestBundleCreate
  firrtl.module @TestBundleCreate() {
    // CHECK: %0 = firrtl.bundlecreate  : () -> !firrtl.bundle<>
    %be = firrtl.bundlecreate : () -> !firrtl.bundle<>

    // CHECK: %c0_ui8 = firrtl.constant 0 : !firrtl.uint<8>
    // CHECK: %c1_ui4 = firrtl.constant 1 : !firrtl.uint<4>
    // %1 = firrtl.bundlecreate %c0_ui8, %c1_ui4 : (!firrtl.uint<8>, !firrtl.uint<4>) -> !firrtl.bundle<a: uint<8>, b: uint<4>>
    %c0 = firrtl.constant 0 : !firrtl.uint<8>
    %c1 = firrtl.constant 1 : !firrtl.uint<4>
    %bc = firrtl.bundlecreate %c0, %c1 : (!firrtl.uint<8>, !firrtl.uint<4>) -> !firrtl.bundle<a: uint<8>, b: uint<4>>

    // %2 = firrtl.aggregateconstant [1, 2, 3, 4] : !firrtl.vector<uint<8>, 4>
    // %3 = firrtl.aggregateconstant [5, 6] : !firrtl.vector<uint<4>, 2>
    // %4 = firrtl.bundlecreate %2, %3 : (!firrtl.vector<uint<8>, 4>, !firrtl.vector<uint<4>, 2>) -> !firrtl.bundle<a: vector<uint<8>, 4>, b: vector<uint<4>, 2>>
    %v0 = firrtl.aggregateconstant [1, 2, 3, 4] : !firrtl.vector<uint<8>, 4>
    %v1 = firrtl.aggregateconstant [5, 6] : !firrtl.vector<uint<4>, 2>
    %bv = firrtl.bundlecreate %v0, %v1 : (!firrtl.vector<uint<8>, 4>, !firrtl.vector<uint<4>, 2>) -> !firrtl.bundle<a: vector<uint<8>, 4>, b: vector<uint<4>, 2>>

    // %5 = firrtl.aggregateconstant [[1, 3], [2, 4]] : !firrtl.bundle<a: vector<uint<8>, 2>, b: vector<uint<5>, 2>>
    // %6 = firrtl.aggregateconstant [[1, 3], [2, 4]] : !firrtl.bundle<a: vector<uint<8>, 2>, b: vector<uint<5>, 2>>
    // %7 = firrtl.bundlecreate %5, %6 : (!firrtl.bundle<a: vector<uint<8>, 2>, b: vector<uint<5>, 2>>, !firrtl.bundle<a: vector<uint<8>, 2>, b: vector<uint<5>, 2>>) -> !firrtl.bundle<a: bundle<a: vector<uint<8>, 2>, b: vector<uint<5>, 2>>, b: bundle<a: vector<uint<8>, 2>, b: vector<uint<5>, 2>>>
    %vb0 = firrtl.aggregateconstant [[1, 2], [3, 4]] : !firrtl.vector<bundle<a: uint<8>, b: uint<5>>, 2>
    %vb1 = firrtl.aggregateconstant [[1, 2], [3, 4]] : !firrtl.vector<bundle<a: uint<8>, b: uint<5>>, 2>
    %bvb = firrtl.bundlecreate %vb0, %vb1 : (!firrtl.vector<bundle<a: uint<8>, b: uint<5>>, 2>, !firrtl.vector<bundle<a: uint<8>, b: uint<5>>, 2>) -> !firrtl.bundle<a: vector<bundle<a: uint<8>, b: uint<5>>, 2>, b: vector<bundle<a: uint<8>, b: uint<5>>, 2>>
  }

  // CHECK-LABEL: @TestVectorCreate
  firrtl.module @TestVectorCreate() {
    // CHECK: %0 = firrtl.vectorcreate  : () -> !firrtl.vector<uint<8>, 0>
    %v0 = firrtl.vectorcreate : () -> !firrtl.vector<uint<8>, 0>

    // CHECK: %c1_ui8 = firrtl.constant 1 : !firrtl.uint<8>
    // CHECK: %1 = firrtl.vectorcreate %c1_ui8, %c1_ui8 : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.vector<uint<8>, 2>
    %c0 = firrtl.constant 1 : !firrtl.uint<8>
    %v1 = firrtl.vectorcreate %c0, %c0: (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.vector<uint<8>, 2>

    // CHECK: %2 = firrtl.bundlecreate %c1_ui8 : (!firrtl.uint<8>) -> !firrtl.bundle<a: uint<8>>
    %b0 = firrtl.bundlecreate %c0 : (!firrtl.uint<8>) -> !firrtl.bundle<a: uint<8>>

    // CHECK: %3 = firrtl.subfield %2[a] : !firrtl.bundle<a: uint<8>>
    // CHECK: %4 = firrtl.vectorcreate %3 : (!firrtl.uint<8>) -> !firrtl.vector<uint<8>, 1>
    // CHECK: %5 = firrtl.bundlecreate %4 : (!firrtl.vector<uint<8>, 1>) -> !firrtl.bundle<a: vector<uint<8>, 1>>
    %v2 = firrtl.vectorcreate %b0 : (!firrtl.bundle<a: uint<8>>) -> !firrtl.vector<bundle<a: uint<8>>, 1>

    // CHECK: %6 = firrtl.bundlecreate %1 : (!firrtl.vector<uint<8>, 2>) -> !firrtl.bundle<a: vector<uint<8>, 2>>
    %b1 = firrtl.bundlecreate %v1 : (!firrtl.vector<uint<8>, 2>) -> !firrtl.bundle<a: vector<uint<8>, 2>>

    // CHECK: %7 = firrtl.subfield %6[a] : !firrtl.bundle<a: vector<uint<8>, 2>>
    // CHECK: %8 = firrtl.vectorcreate %7 : (!firrtl.vector<uint<8>, 2>) -> !firrtl.vector<vector<uint<8>, 2>, 1>
    // CHECK: %9 = firrtl.bundlecreate %8 : (!firrtl.vector<vector<uint<8>, 2>, 1>) -> !firrtl.bundle<a: vector<vector<uint<8>, 2>, 1>>
    %v3 = firrtl.vectorcreate %b1 : (!firrtl.bundle<a: vector<uint<8>, 2>>) -> !firrtl.vector<bundle<a: vector<uint<8>, 2>>, 1>
  }

    // CHECK-LABEL: @TestVBAggregate
  firrtl.module @TestVBAggregate() {
    // CHECK: %0 = firrtl.aggregateconstant [1, 2] : !firrtl.bundle<a: uint<8>, b: uint<5>>
    // CHECK: %1 = firrtl.subfield %0[b] : !firrtl.bundle<a: uint<8>, b: uint<5>>
    // CHECK: %2 = firrtl.subfield %0[a] : !firrtl.bundle<a: uint<8>, b: uint<5>>
    %0 = firrtl.aggregateconstant [1, 2] : !firrtl.bundle<a: uint<8>, b: uint<5>>

    // CHECK: %3 = firrtl.aggregateconstant [3, 4] : !firrtl.bundle<a: uint<8>, b: uint<5>>
    // CHECK: %4 = firrtl.subfield %3[b] : !firrtl.bundle<a: uint<8>, b: uint<5>>
    // CHECK: %5 = firrtl.subfield %3[a] : !firrtl.bundle<a: uint<8>, b: uint<5>>
    %1 = firrtl.aggregateconstant [3, 4] : !firrtl.bundle<a: uint<8>, b: uint<5>>

    // CHECK: %6 = firrtl.vectorcreate %2, %5 : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.vector<uint<8>, 2>
    // CHECK: %7 = firrtl.vectorcreate %1, %4 : (!firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.vector<uint<5>, 2>
    // CHECK: %8 = firrtl.bundlecreate %6, %7 : (!firrtl.vector<uint<8>, 2>, !firrtl.vector<uint<5>, 2>) -> !firrtl.bundle<a: vector<uint<8>, 2>, b: vector<uint<5>, 2>>
    %2 = firrtl.vectorcreate %0, %1 : (!firrtl.bundle<a: uint<8>, b: uint<5>>, !firrtl.bundle<a: uint<8>, b: uint<5>>) -> !firrtl.vector<bundle<a: uint<8>, b: uint<5>>, 2>
  }

  firrtl.module @TestVVBAggregate() {
    // CHECK{LITERAL}: %0 = firrtl.aggregateconstant [[1, 3], [2, 4]] : !firrtl.bundle<a: vector<uint<8>, 2>, b: vector<uint<5>, 2>>
    // CHECK: %1 = firrtl.subfield %0[b] : !firrtl.bundle<a: vector<uint<8>, 2>, b: vector<uint<5>, 2>>
    // CHECK: %2 = firrtl.subfield %0[a] : !firrtl.bundle<a: vector<uint<8>, 2>, b: vector<uint<5>, 2>>
    %0 = firrtl.aggregateconstant [[1, 2], [3, 4]] : !firrtl.vector<bundle<a: uint<8>, b: uint<5>>, 2>
    
    // CHECK{LITERAL}: %3 = firrtl.aggregateconstant [[5, 7], [6, 8]] : !firrtl.bundle<a: vector<uint<8>, 2>, b: vector<uint<5>, 2>>
    // CHECK: %4 = firrtl.subfield %3[b] : !firrtl.bundle<a: vector<uint<8>, 2>, b: vector<uint<5>, 2>>
    // CHECK: %5 = firrtl.subfield %3[a] : !firrtl.bundle<a: vector<uint<8>, 2>, b: vector<uint<5>, 2>>
    %1 = firrtl.aggregateconstant [[5, 6], [7, 8]] : !firrtl.vector<bundle<a: uint<8>, b: uint<5>>, 2>

    // CHECK: %6 = firrtl.vectorcreate %2, %5 : (!firrtl.vector<uint<8>, 2>, !firrtl.vector<uint<8>, 2>) -> !firrtl.vector<vector<uint<8>, 2>, 2>
    // CHECK: %7 = firrtl.vectorcreate %1, %4 : (!firrtl.vector<uint<5>, 2>, !firrtl.vector<uint<5>, 2>) -> !firrtl.vector<vector<uint<5>, 2>, 2>
    // CHECK: %8 = firrtl.bundlecreate %6, %7 : (!firrtl.vector<vector<uint<8>, 2>, 2>, !firrtl.vector<vector<uint<5>, 2>, 2>) -> !firrtl.bundle<a: vector<vector<uint<8>, 2>, 2>, b: vector<vector<uint<5>, 2>, 2>>
    %2 = firrtl.vectorcreate %0, %1 : (!firrtl.vector<bundle<a: uint<8>, b: uint<5>>, 2>, !firrtl.vector<bundle<a: uint<8>, b: uint<5>>, 2>) -> !firrtl.vector<vector<bundle<a: uint<8>, b: uint<5>>, 2>, 2>
  }

  //===--------------------------------------------------------------------===//
  // Declaration Tests
  //===--------------------------------------------------------------------===//

  // CHECK-LABEL: @TestWire
  firrtl.module @TestWire() {
    // CHECK: %0 = firrtl.wire : !firrtl.uint<8>
    %0 = firrtl.wire : !firrtl.uint<8>
    // CHECK: %1 = firrtl.wire : !firrtl.bundle<>
    %1 = firrtl.wire : !firrtl.bundle<>
    // CHECK: %2 = firrtl.wire : !firrtl.bundle<a: uint<8>>
    %2 = firrtl.wire : !firrtl.bundle<a: uint<8>>
    // CHECK: %3 = firrtl.wire : !firrtl.vector<uint<8>, 0>
    %3 = firrtl.wire : !firrtl.vector<uint<8>, 0>
    // CHECK: %4 = firrtl.wire : !firrtl.vector<uint<8>, 2>
    %4 = firrtl.wire : !firrtl.vector<uint<8>, 2>
    // CHECK: %5 = firrtl.wire : !firrtl.bundle<a: vector<uint<8>, 2>
    %5 = firrtl.wire : !firrtl.bundle<a: vector<uint<8>, 2>>
    // CHECK: %6 = firrtl.wire : !firrtl.bundle<a: vector<uint<8>, 2>>
    %6 = firrtl.wire : !firrtl.vector<bundle<a: uint<8>>, 2>
    // CHECK: %7 = firrtl.wire : !firrtl.bundle<a: bundle<b: vector<uint<8>, 2>>>
    %7 = firrtl.wire : !firrtl.bundle<a: vector<bundle<b: uint<8>>, 2>>
  }

  // CHECK-LABEL: @TestNode
  firrtl.module @TestNode() {
    // CHECK: %w = firrtl.wire : !firrtl.bundle<a: vector<uint<8>, 2>>
    %w = firrtl.wire : !firrtl.vector<bundle<a: uint<8>>, 2>
    // CHECK: %n = firrtl.node %w : !firrtl.bundle<a: vector<uint<8>, 2>>
    %n = firrtl.node %w : !firrtl.vector<bundle<a: uint<8>>, 2>
  }

  // CHECK-LABEL: @TestNodeMaterializedFromExplodedBundle
  firrtl.module @TestNodeMaterializedFromExplodedBundle() {
    // CHECK: %w = firrtl.wire : !firrtl.bundle<a: vector<uint<8>, 2>>
    // CHECK: %0 = firrtl.subfield %w[a] : !firrtl.bundle<a: vector<uint<8>, 2>>
    // CHECK: %1 = firrtl.subindex %0[0] : !firrtl.vector<uint<8>, 2>
    // CHECK: %2 = firrtl.bundlecreate %1 : (!firrtl.uint<8>) -> !firrtl.bundle<a: uint<8>>
    // CHECK: %m = firrtl.node %2 : !firrtl.bundle<a: uint<8>>
    %w = firrtl.wire : !firrtl.vector<bundle<a: uint<8>>, 2>
    %b = firrtl.subindex %w[0] : !firrtl.vector<bundle<a: uint<8>>, 2>
    %m = firrtl.node %b : !firrtl.bundle<a: uint<8>>
  }
  
  // CHECK-LABEL: @TestReg
  firrtl.module @TestReg(in %clock: !firrtl.clock) {
    // CHECK: %r = firrtl.reg %clock : !firrtl.clock, !firrtl.bundle<a: vector<uint<8>, 2>>
    %r = firrtl.reg %clock : !firrtl.clock, !firrtl.vector<bundle<a: uint<8>>, 2>
  }

  // CHECK-LABEL: @TestRegReset
  firrtl.module @TestRegReset(in %clock: !firrtl.clock) {
    // CHECK{LITERAL}: %0 = firrtl.aggregateconstant [[1, 3], [2, 4]] : !firrtl.bundle<a: vector<uint<4>, 2>, b: vector<uint<8>, 2>>
     %rval = firrtl.aggregateconstant [[1, 2], [3, 4]] : !firrtl.vector<bundle<a: uint<4>, b: uint<8>>, 2> 
    // CHECK: %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %rsig = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK: %r = firrtl.regreset %clock, %c0_ui1, %0 : !firrtl.clock, !firrtl.uint<1>, !firrtl.bundle<a: vector<uint<4>, 2>, b: vector<uint<8>, 2>>, !firrtl.bundle<a: vector<uint<4>, 2>, b: vector<uint<8>, 2>>
    %r = firrtl.regreset %clock, %rsig, %rval : !firrtl.clock, !firrtl.uint<1>, !firrtl.vector<bundle<a: uint<4>, b: uint<8>>, 2>, !firrtl.vector<bundle<a: uint<4>, b: uint<8>>, 2>
  }

  // CHECK-LABEL: @TestRegResetMaterializedFromExplodedBundle
  firrtl.module @TestRegResetMaterializedFromExplodedBundle(in %clock: !firrtl.clock) {
    // CHECK{LITERAL}: %0 = firrtl.aggregateconstant [[1, 3], [2, 4]] : !firrtl.bundle<a: vector<uint<4>, 2>, b: vector<uint<8>, 2>>
    %storage = firrtl.aggregateconstant [[1, 2], [3, 4]] : !firrtl.vector<bundle<a: uint<4>, b: uint<8>>, 2> 
    // CHECK: %1 = firrtl.subfield %0[b] : !firrtl.bundle<a: vector<uint<4>, 2>, b: vector<uint<8>, 2>>
    // CHECK: %2 = firrtl.subindex %1[0] : !firrtl.vector<uint<8>, 2>
    // CHECK: %3 = firrtl.subfield %0[a] : !firrtl.bundle<a: vector<uint<4>, 2>, b: vector<uint<8>, 2>>
    // CHECK: %4 = firrtl.subindex %3[0] : !firrtl.vector<uint<4>, 2>
    // CHECK: %5 = firrtl.bundlecreate %4, %2 : (!firrtl.uint<4>, !firrtl.uint<8>) -> !firrtl.bundle<a: uint<4>, b: uint<8>>
    %rval = firrtl.subindex %storage[0] : !firrtl.vector<bundle<a: uint<4>, b: uint<8>>, 2>
    // CHECK: %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %rsig = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK: %r = firrtl.regreset %clock, %c0_ui1, %5 : !firrtl.clock, !firrtl.uint<1>, !firrtl.bundle<a: uint<4>, b: uint<8>>, !firrtl.bundle<a: uint<4>, b: uint<8>>
    %r = firrtl.regreset %clock, %rsig, %rval : !firrtl.clock, !firrtl.uint<1>, !firrtl.bundle<a: uint<4>, b: uint<8>>, !firrtl.bundle<a: uint<4>, b: uint<8>>
  }

  // CHECK-LABEL: @TestRegResetMaterializedFromDeepExplodedBundle
  firrtl.module @TestRegResetMaterializedFromDeepExplodedBundle(in %clock: !firrtl.clock) {
    // CHECK{LITERAL}: %0 = firrtl.aggregateconstant [[1, 4], [[2, 3], [5, 6]]] : !firrtl.bundle<a: vector<uint<4>, 2>, b: bundle<c: vector<uint<8>, 2>, d: vector<uint<16>, 2>>>
    // CHECK: %1 = firrtl.subfield %0[b] : !firrtl.bundle<a: vector<uint<4>, 2>, b: bundle<c: vector<uint<8>, 2>, d: vector<uint<16>, 2>>>
    // CHECK: %2 = firrtl.subfield %1[d] : !firrtl.bundle<c: vector<uint<8>, 2>, d: vector<uint<16>, 2>>
    // CHECK: %3 = firrtl.subindex %2[1] : !firrtl.vector<uint<16>, 2>
    // CHECK: %4 = firrtl.subfield %1[c] : !firrtl.bundle<c: vector<uint<8>, 2>, d: vector<uint<16>, 2>>
    // CHECK: %5 = firrtl.subindex %4[1] : !firrtl.vector<uint<8>, 2>
    // CHECK: %6 = firrtl.subfield %0[a] : !firrtl.bundle<a: vector<uint<4>, 2>, b: bundle<c: vector<uint<8>, 2>, d: vector<uint<16>, 2>>>
    // CHECK: %7 = firrtl.subindex %6[1] : !firrtl.vector<uint<4>, 2>
    // CHECK: %8 = firrtl.bundlecreate %5, %3 : (!firrtl.uint<8>, !firrtl.uint<16>) -> !firrtl.bundle<c: uint<8>, d: uint<16>>
    // CHECK: %9 = firrtl.bundlecreate %7, %8 : (!firrtl.uint<4>, !firrtl.bundle<c: uint<8>, d: uint<16>>) -> !firrtl.bundle<a: uint<4>, b: bundle<c: uint<8>, d: uint<16>>>
    // CHECK: %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK: %reg = firrtl.regreset %clock, %c0_ui1, %9 : !firrtl.clock, !firrtl.uint<1>, !firrtl.bundle<a: uint<4>, b: bundle<c: uint<8>, d: uint<16>>>, !firrtl.bundle<a: uint<4>, b: bundle<c: uint<8>, d: uint<16>>>
    %reset_value_storage = firrtl.aggregateconstant [[1, [2, 3]], [4, [5, 6]]] : !firrtl.vector<bundle<a: uint<4>, b: bundle<c: uint<8>, d: uint<16>>>, 2> 
    %reset_value = firrtl.subindex %reset_value_storage[1] : !firrtl.vector<bundle<a: uint<4>, b: bundle<c: uint<8>, d: uint<16>>>, 2>
    %reset = firrtl.constant 0 : !firrtl.uint<1>
    %reg = firrtl.regreset %clock, %reset, %reset_value : !firrtl.clock, !firrtl.uint<1>, !firrtl.bundle<a: uint<4>, b: bundle<c: uint<8>, d: uint<16>>>, !firrtl.bundle<a: uint<4>, b: bundle<c: uint<8>, d: uint<16>>>
  }

  // CHECK-LABEL: @TestInstance
  firrtl.module @TestInstance() {
    // CHECK: %myinst_port = firrtl.instance myinst @VB(in port: !firrtl.bundle<a: vector<uint<8>, 4>>)
    %myinst_port = firrtl.instance myinst @VB(in port: !firrtl.vector<bundle<a: uint<8>>, 4>)
  }

  //===--------------------------------------------------------------------===//
  // Connect Tests
  //===--------------------------------------------------------------------===//

  // CHECK-LABEL: @TestBasicConnects
  firrtl.module @TestBasicConnects() {
    // CHECK: %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    // CHECK: %w0 = firrtl.wire : !firrtl.uint<1>
    // CHECK: firrtl.strictconnect %w0, %c1_ui1 : !firrtl.uint<1>
    %c1 = firrtl.constant 1 : !firrtl.uint<1>
    %w0 = firrtl.wire : !firrtl.uint<1>
    firrtl.strictconnect %w0, %c1 : !firrtl.uint<1>

    // CHECK: %w1 = firrtl.wire : !firrtl.bundle<a: bundle<>>
    // CHECK: %w2 = firrtl.wire : !firrtl.bundle<a: bundle<>>
    // CHECK: firrtl.strictconnect %w1, %w2 : !firrtl.bundle<a: bundle<>>
    %w1 = firrtl.wire : !firrtl.bundle<a: bundle<>>
    %w2 = firrtl.wire : !firrtl.bundle<a: bundle<>>
    firrtl.strictconnect %w1, %w2 : !firrtl.bundle<a: bundle<>>

    // CHECK: %w3 = firrtl.wire : !firrtl.bundle<a flip: uint<1>>
    // CHECK: %w4 = firrtl.wire : !firrtl.bundle<a flip: uint<1>>
    // CHECK: firrtl.connect %w3, %w4 : !firrtl.bundle<a flip: uint<1>>
    %w3 = firrtl.wire : !firrtl.bundle<a flip: uint<1>>
    %w4 = firrtl.wire : !firrtl.bundle<a flip: uint<1>>
    firrtl.connect %w3, %w4 : !firrtl.bundle<a flip: uint<1>>, !firrtl.bundle<a flip: uint<1>>

    // CHECK: %w5 = firrtl.wire : !firrtl.bundle<a flip: uint<1>, b: uint<1>>
    // CHECK: %w6 = firrtl.wire : !firrtl.bundle<a flip: uint<1>, b: uint<1>>
    // CHECK: firrtl.connect %w5, %w6 : !firrtl.bundle<a flip: uint<1>, b: uint<1>>
    %w5 = firrtl.wire : !firrtl.bundle<a flip: uint<1>, b: uint<1>>
    %w6 = firrtl.wire : !firrtl.bundle<a flip: uint<1>, b: uint<1>>
    firrtl.connect %w5, %w6 : !firrtl.bundle<a flip: uint<1>, b: uint<1>>, !firrtl.bundle<a flip: uint<1>, b: uint<1>>
  
    // CHECK: %w7 = firrtl.wire : !firrtl.bundle<a: bundle<b flip: uint<8>>>
    // CHECK: %w8 = firrtl.wire : !firrtl.bundle<a: bundle<b flip: uint<8>>>
    // CHECK: firrtl.connect %w7, %w8 : !firrtl.bundle<a: bundle<b flip: uint<8>>>
    %w7 = firrtl.wire : !firrtl.bundle<a: bundle<b flip: uint<8>>>
    %w8 = firrtl.wire : !firrtl.bundle<a: bundle<b flip: uint<8>>>
    firrtl.connect %w7, %w8 : !firrtl.bundle<a: bundle<b flip: uint<8>>>, !firrtl.bundle<a: bundle<b flip: uint<8>>>
  
    // Test some deeper connections.
    // (access-path caching causes subfield/subindex op movement)
  
    // CHECK: %w9 = firrtl.wire : !firrtl.bundle<a flip: bundle<b flip: uint<8>>>
    // CHECK: %0 = firrtl.subfield %w9[a] : !firrtl.bundle<a flip: bundle<b flip: uint<8>>>
    // CHECK: %1 = firrtl.subfield %0[b] : !firrtl.bundle<b flip: uint<8>>
    %w9  = firrtl.wire : !firrtl.bundle<a flip: bundle<b flip: uint<8>>>
  
    // CHECK: %w10 = firrtl.wire : !firrtl.bundle<a flip: bundle<b flip: uint<8>>>
    // CHECK: %2 = firrtl.subfield %w10[a] : !firrtl.bundle<a flip: bundle<b flip: uint<8>>>
    // CHECK: %3 = firrtl.subfield %2[b] : !firrtl.bundle<b flip: uint<8>>
    %w10 = firrtl.wire : !firrtl.bundle<a flip: bundle<b flip: uint<8>>>

    // CHECK: firrtl.connect %w9, %w10 : !firrtl.bundle<a flip: bundle<b flip: uint<8>>>
    firrtl.connect %w9, %w10 : !firrtl.bundle<a flip: bundle<b flip: uint<8>>>, !firrtl.bundle<a flip: bundle<b flip: uint<8>>>

    %w9_a = firrtl.subfield %w9[a] : !firrtl.bundle<a flip: bundle<b flip: uint<8>>>
    %w10_a = firrtl.subfield %w10[a] : !firrtl.bundle<a flip: bundle<b flip: uint<8>>>
    
    // CHECK: firrtl.connect %0, %2 : !firrtl.bundle<b flip: uint<8>>
    firrtl.connect %w9_a, %w10_a: !firrtl.bundle<b flip: uint<8>>, !firrtl.bundle<b flip: uint<8>>
  
    %w9_a_b = firrtl.subfield %w9_a[b] : !firrtl.bundle<b flip: uint<8>>
    %w10_a_b = firrtl.subfield %w10_a[b] : !firrtl.bundle<b flip: uint<8>>

    // CHECK: firrtl.strictconnect %1, %3 : !firrtl.uint<8>
    firrtl.strictconnect %w9_a_b, %w10_a_b : !firrtl.uint<8>
  }

  //===--------------------------------------------------------------------===//
  // Path Tests
  //===--------------------------------------------------------------------===//

  // CHECK-LABEL: @TestSubfield
  firrtl.module  @TestSubfield() {
    // CHECK: %b1 = firrtl.wire : !firrtl.bundle<a: uint<1>>
    // CHECK: %0 = firrtl.subfield %b1[a] : !firrtl.bundle<a: uint<1>>
    // CHECK: %b2 = firrtl.wire : !firrtl.bundle<b: uint<1>>
    // CHECK: %1 = firrtl.subfield %b2[b] : !firrtl.bundle<b: uint<1>>
    // CHECK: firrtl.strictconnect %0, %1 : !firrtl.uint<1>
    %b1 = firrtl.wire : !firrtl.bundle<a: uint<1>>
    %b2 = firrtl.wire : !firrtl.bundle<b: uint<1>>
    %a = firrtl.subfield %b1[a] : !firrtl.bundle<a: uint<1>>
    %b = firrtl.subfield %b2[b] : !firrtl.bundle<b: uint<1>>
    firrtl.strictconnect %a, %b : !firrtl.uint<1>
  }

  // CHECK-LABEL: @TestSubindex
  firrtl.module @TestSubindex(
    // CHECK-SAME: in %port: !firrtl.bundle<a flip: vector<uint<8>, 4>>
    in %port: !firrtl.vector<bundle<a flip: uint<8>>, 4>) {

    // Basic test that a path is rewritten following a vb->bv conversion
  
    // CHECK: %0 = firrtl.subfield %port[a] : !firrtl.bundle<a flip: vector<uint<8>, 4>>
    // CHECK: %1 = firrtl.subindex %0[3] : !firrtl.vector<uint<8>, 4>
    // CHECK: %c7_ui8 = firrtl.constant 7 : !firrtl.uint<8>
    // CHECK: firrtl.connect %1, %c7_ui8 : !firrtl.uint<8>, !firrtl.uint<8>
    %bundle = firrtl.subindex %port[3] : !firrtl.vector<bundle<a flip: uint<8>>, 4>
    %field  = firrtl.subfield %bundle[a] : !firrtl.bundle<a flip: uint<8>>
    %value  = firrtl.constant 7 : !firrtl.uint<8>
    firrtl.connect %field, %value : !firrtl.uint<8>, !firrtl.uint<8>

    // Connect two exploded bundles.
  
    // CHECK: %v1 = firrtl.wire : !firrtl.bundle<a: vector<vector<uint<8>, 8>, 2>>
    // CHECK: %2 = firrtl.subfield %v1[a] : !firrtl.bundle<a: vector<vector<uint<8>, 8>, 2>>
    // CHECK: %3 = firrtl.subindex %2[0] : !firrtl.vector<vector<uint<8>, 8>, 2>
    %v1 = firrtl.wire : !firrtl.vector<bundle<a: vector<uint<8>, 8>>, 2>
  
    // CHECK: %v2 = firrtl.wire : !firrtl.bundle<a: vector<vector<uint<8>, 8>, 2>>
    // CHECK: %4 = firrtl.subfield %v2[a] : !firrtl.bundle<a: vector<vector<uint<8>, 8>, 2>>
    // CHECK: %5 = firrtl.subindex %4[0] : !firrtl.vector<vector<uint<8>, 8>, 2>
    %v2 = firrtl.wire : !firrtl.vector<bundle<a: vector<uint<8>, 8>>, 2>
  
    %b1 = firrtl.subindex %v1[0] : !firrtl.vector<bundle<a: vector<uint<8>, 8>>, 2>
    %b2 = firrtl.subindex %v2[0] : !firrtl.vector<bundle<a: vector<uint<8>, 8>>, 2>
  
    // CHECK: firrtl.strictconnect %3, %5 : !firrtl.vector<uint<8>, 8>
    firrtl.strictconnect %b1, %b2 : !firrtl.bundle<a: vector<uint<8>, 8>>
  }

  // CHECK-LABEL: TestSubaccess
  firrtl.module @TestSubaccess() {
    // CHECK: %c0_ui8 = firrtl.constant 0 : !firrtl.uint<8>
    %0 = firrtl.constant 0 : !firrtl.uint<8>
    // CHECK: %0 = firrtl.vectorcreate %c0_ui8, %c0_ui8 : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.vector<uint<8>, 2>
    %v = firrtl.vectorcreate %0, %0 : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.vector<uint<8>, 2>
    // CHECK: %dst = firrtl.wire : !firrtl.uint<8>
    %dst = firrtl.wire : !firrtl.uint<8>
    // CHECK: %1 = firrtl.subaccess %0[%c0_ui8] : !firrtl.vector<uint<8>, 2>, !firrtl.uint<8>
    %src = firrtl.subaccess %v[%0] : !firrtl.vector<uint<8>, 2>, !firrtl.uint<8>
    // CHECK: firrtl.strictconnect %dst, %1 : !firrtl.uint<8>
    firrtl.strictconnect %dst, %src : !firrtl.uint<8>
  }

  // CHECK-LABEL: TestSubaccess2
  firrtl.module @TestSubaccess2() {
    // CHECK: %c0_ui8 = firrtl.constant 0 : !firrtl.uint<8>
    %0 = firrtl.constant 0 : !firrtl.uint<8>
    // CHECK: %0 = firrtl.vectorcreate %c0_ui8, %c0_ui8 : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.vector<uint<8>, 2>
    %v = firrtl.vectorcreate %0, %0 : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.vector<uint<8>, 2>
    // CHECK: %c0_ui8_0 = firrtl.constant 0 : !firrtl.uint<8>
    %i = firrtl.constant 0 : !firrtl.uint<8>
    // CHECK: %dst = firrtl.wire : !firrtl.uint<8>
    %dst = firrtl.wire : !firrtl.uint<8>
    // CHECK: %1 = firrtl.subaccess %0[%c0_ui8_0] : !firrtl.vector<uint<8>, 2>, !firrtl.uint<8>
    %src = firrtl.subaccess %v[%i] : !firrtl.vector<uint<8>, 2>, !firrtl.uint<8>
    // CHECK: firrtl.strictconnect %dst, %1 : !firrtl.uint<8>
    firrtl.strictconnect %dst, %src : !firrtl.uint<8>
  }

  // CHECK-LABEL: @TestPathCaching()
  firrtl.module @TestPathCaching() {
    // CHECK: %w = firrtl.wire : !firrtl.bundle<a: bundle<b: uint<8>>>
    // CHECK: %0 = firrtl.subfield %w[a] : !firrtl.bundle<a: bundle<b: uint<8>>>
    // CHECK: %n1 = firrtl.node %0 : !firrtl.bundle<b: uint<8>>
    // CHECK: %n2 = firrtl.node %0 :  !firrtl.bundle<b: uint<8>>
    %w = firrtl.wire : !firrtl.bundle<a: bundle<b: uint<8>>>
    %a1 = firrtl.subfield %w[a] : !firrtl.bundle<a: bundle<b: uint<8>>>
    %n1 = firrtl.node %a1 : !firrtl.bundle<b: uint<8>>
    %a2 = firrtl.subfield %w[a] : !firrtl.bundle<a: bundle<b: uint<8>>>
    %n2 = firrtl.node %a2 : !firrtl.bundle<b: uint<8>>
  }

  //===--------------------------------------------------------------------===//
  // Operand Explosion Tests
  //===--------------------------------------------------------------------===//

  // CHECK-LABEL: @TestLhsExploded
  firrtl.module @TestLhsExploded() {
    // CHECK: %lhs_storage = firrtl.wire : !firrtl.bundle<a: vector<uint<1>, 2>, b: bundle<c: vector<uint<1>, 2>, d: vector<uint<1>, 2>>>
    // CHECK: %0 = firrtl.subfield %lhs_storage[b] : !firrtl.bundle<a: vector<uint<1>, 2>, b: bundle<c: vector<uint<1>, 2>, d: vector<uint<1>, 2>>>
    // CHECK: %1 = firrtl.subfield %0[d] : !firrtl.bundle<c: vector<uint<1>, 2>, d: vector<uint<1>, 2>>
    // CHECK: %2 = firrtl.subindex %1[0] : !firrtl.vector<uint<1>, 2>
    // CHECK: %3 = firrtl.subfield %0[c] : !firrtl.bundle<c: vector<uint<1>, 2>, d: vector<uint<1>, 2>>
    // CHECK: %4 = firrtl.subindex %3[0] : !firrtl.vector<uint<1>, 2>
    // CHECK: %5 = firrtl.subfield %lhs_storage[a] : !firrtl.bundle<a: vector<uint<1>, 2>, b: bundle<c: vector<uint<1>, 2>, d: vector<uint<1>, 2>>>
    // CHECK: %6 = firrtl.subindex %5[0] : !firrtl.vector<uint<1>, 2>
    // CHECK: %rhs = firrtl.wire : !firrtl.bundle<a: uint<1>, b: bundle<c: uint<1>, d: uint<1>>>
    // CHECK: %7 = firrtl.subfield %rhs[b] : !firrtl.bundle<a: uint<1>, b: bundle<c: uint<1>, d: uint<1>>>
    // CHECK: %8 = firrtl.subfield %7[d] : !firrtl.bundle<c: uint<1>, d: uint<1>>
    // CHECK: %9 = firrtl.subfield %7[c] : !firrtl.bundle<c: uint<1>, d: uint<1>>
    // CHECK: %10 = firrtl.subfield %rhs[a] : !firrtl.bundle<a: uint<1>, b: bundle<c: uint<1>, d: uint<1>>>
    // CHECK: firrtl.strictconnect %6, %10 : !firrtl.uint<1>
    // CHECK: firrtl.strictconnect %4, %9 : !firrtl.uint<1>
    // CHECK: firrtl.strictconnect %2, %8 : !firrtl.uint<1>
    %lhs_storage  = firrtl.wire : !firrtl.vector<bundle<a: uint<1>, b: bundle<c: uint<1>, d: uint<1>>>, 2>
    %lhs = firrtl.subindex %lhs_storage[0] : !firrtl.vector<bundle<a: uint<1>, b: bundle<c: uint<1>, d: uint<1>>>, 2>
    %rhs = firrtl.wire : !firrtl.bundle<a: uint<1>, b: bundle<c: uint<1>, d: uint<1>>>
    firrtl.strictconnect %lhs, %rhs : !firrtl.bundle<a: uint<1>, b: bundle<c: uint<1>, d: uint<1>>>
  }

  // CHECK-LABEL: @TestLhsExplodedWhenLhsHasFlips
  firrtl.module @TestLhsExplodedWhenLhsHasFlips() {
    // CHECK: %lhs_storage = firrtl.wire : !firrtl.bundle<a flip: vector<uint<1>, 2>, b: vector<uint<1>, 2>>
    // CHECK: %0 = firrtl.subfield %lhs_storage[b] : !firrtl.bundle<a flip: vector<uint<1>, 2>, b: vector<uint<1>, 2>>
    // CHECK: %1 = firrtl.subindex %0[0] : !firrtl.vector<uint<1>, 2>
    // CHECK: %2 = firrtl.subfield %lhs_storage[a] : !firrtl.bundle<a flip: vector<uint<1>, 2>, b: vector<uint<1>, 2>>
    // CHECK: %3 = firrtl.subindex %2[0] : !firrtl.vector<uint<1>, 2>
    // CHECK: %rhs = firrtl.wire : !firrtl.bundle<a flip: uint<1>, b: uint<1>>
    // CHECK: %4 = firrtl.subfield %rhs[b] : !firrtl.bundle<a flip: uint<1>, b: uint<1>>
    // CHECK: %5 = firrtl.subfield %rhs[a] : !firrtl.bundle<a flip: uint<1>, b: uint<1>>
    // CHECK: firrtl.connect %5, %3 : !firrtl.uint<1>
    // CHECK: firrtl.connect %1, %4 : !firrtl.uint<1>
    %lhs_storage = firrtl.wire : !firrtl.vector<bundle<a flip: uint<1>, b: uint<1>>, 2>
    %lhs = firrtl.subindex %lhs_storage[0] : !firrtl.vector<bundle<a flip: uint<1>, b: uint<1>>, 2>
    %rhs = firrtl.wire : !firrtl.bundle<a flip: uint<1>, b: uint<1>>
    firrtl.connect %lhs, %rhs : !firrtl.bundle<a flip: uint<1>, b: uint<1>>, !firrtl.bundle<a flip: uint<1>, b: uint<1>>
  }

  // CHECK-LABEL: @TestRhsExploded
  firrtl.module @TestRhsExploded() {
    // CHECK: %lhs = firrtl.wire : !firrtl.bundle<a: uint<1>, b: bundle<c: uint<1>, d: uint<1>>>
    // CHECK: %rhs_storage = firrtl.wire : !firrtl.bundle<a: vector<uint<1>, 2>, b: bundle<c: vector<uint<1>, 2>, d: vector<uint<1>, 2>>>
    // CHECK: %0 = firrtl.subfield %rhs_storage[b] : !firrtl.bundle<a: vector<uint<1>, 2>, b: bundle<c: vector<uint<1>, 2>, d: vector<uint<1>, 2>>>
    // CHECK: %1 = firrtl.subfield %0[d] : !firrtl.bundle<c: vector<uint<1>, 2>, d: vector<uint<1>, 2>>
    // CHECK: %2 = firrtl.subindex %1[0] : !firrtl.vector<uint<1>, 2>
    // CHECK: %3 = firrtl.subfield %0[c] : !firrtl.bundle<c: vector<uint<1>, 2>, d: vector<uint<1>, 2>>
    // CHECK: %4 = firrtl.subindex %3[0] : !firrtl.vector<uint<1>, 2>
    // CHECK: %5 = firrtl.subfield %rhs_storage[a] : !firrtl.bundle<a: vector<uint<1>, 2>, b: bundle<c: vector<uint<1>, 2>, d: vector<uint<1>, 2>>>
    // CHECK: %6 = firrtl.subindex %5[0] : !firrtl.vector<uint<1>, 2>
    // CHECK: %7 = firrtl.bundlecreate %4, %2 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.bundle<c: uint<1>, d: uint<1>>
    // CHECK: %8 = firrtl.bundlecreate %6, %7 : (!firrtl.uint<1>, !firrtl.bundle<c: uint<1>, d: uint<1>>) -> !firrtl.bundle<a: uint<1>, b: bundle<c: uint<1>, d: uint<1>>>
    // CHECK: firrtl.strictconnect %lhs, %8 : !firrtl.bundle<a: uint<1>, b: bundle<c: uint<1>, d: uint<1>>>
    %lhs = firrtl.wire : !firrtl.bundle<a: uint<1>, b: bundle<c: uint<1>, d: uint<1>>>
    %rhs_storage = firrtl.wire : !firrtl.vector<bundle<a: uint<1>, b: bundle<c: uint<1>, d: uint<1>>>, 2>
    %rhs = firrtl.subindex %rhs_storage[0] : !firrtl.vector<bundle<a: uint<1>, b: bundle<c: uint<1>, d: uint<1>>>, 2>
    firrtl.strictconnect %lhs, %rhs : !firrtl.bundle<a: uint<1>, b: bundle<c: uint<1>, d: uint<1>>>
  }

  // CHECK-LABEL: @TestRhsExplodedWhenLhsHasFlips
  firrtl.module @TestRhsExplodedWhenLhsHasFlips() {
    // CHECK: %lhs = firrtl.wire : !firrtl.bundle<a: uint<1>, b: bundle<c: uint<1>, d flip: uint<1>>>
    // CHECK: %0 = firrtl.subfield %lhs[b] : !firrtl.bundle<a: uint<1>, b: bundle<c: uint<1>, d flip: uint<1>>>
    // CHECK: %1 = firrtl.subfield %0[d] : !firrtl.bundle<c: uint<1>, d flip: uint<1>>
    // CHECK: %2 = firrtl.subfield %0[c] : !firrtl.bundle<c: uint<1>, d flip: uint<1>>
    // CHECK: %3 = firrtl.subfield %lhs[a] : !firrtl.bundle<a: uint<1>, b: bundle<c: uint<1>, d flip: uint<1>>>
    // CHECK: %rhs_storage = firrtl.wire : !firrtl.bundle<a: vector<uint<1>, 2>, b: bundle<c: vector<uint<1>, 2>, d flip: vector<uint<1>, 2>>>
    // CHECK: %4 = firrtl.subfield %rhs_storage[b] : !firrtl.bundle<a: vector<uint<1>, 2>, b: bundle<c: vector<uint<1>, 2>, d flip: vector<uint<1>, 2>>>
    // CHECK: %5 = firrtl.subfield %4[d] : !firrtl.bundle<c: vector<uint<1>, 2>, d flip: vector<uint<1>, 2>>
    // CHECK: %6 = firrtl.subindex %5[0] : !firrtl.vector<uint<1>, 2>
    // CHECK: %7 = firrtl.subfield %4[c] : !firrtl.bundle<c: vector<uint<1>, 2>, d flip: vector<uint<1>, 2>>
    // CHECK: %8 = firrtl.subindex %7[0] : !firrtl.vector<uint<1>, 2>
    // CHECK: %9 = firrtl.subfield %rhs_storage[a] : !firrtl.bundle<a: vector<uint<1>, 2>, b: bundle<c: vector<uint<1>, 2>, d flip: vector<uint<1>, 2>>>
    // CHECK: %10 = firrtl.subindex %9[0] : !firrtl.vector<uint<1>, 2>
    // CHECK: firrtl.connect %3, %10 : !firrtl.uint<1>
    // CHECK: firrtl.connect %2, %8 : !firrtl.uint<1>
    // CHECK: firrtl.connect %6, %1 : !firrtl.uint<1>
    %lhs = firrtl.wire : !firrtl.bundle<a: uint<1>, b: bundle<c: uint<1>, d flip: uint<1>>>
    %rhs_storage = firrtl.wire : !firrtl.vector<bundle<a: uint<1>, b: bundle<c: uint<1>, d flip: uint<1>>>, 2>
    %rhs = firrtl.subindex %rhs_storage[0] : !firrtl.vector<bundle<a: uint<1>, b: bundle<c: uint<1>, d flip: uint<1>>>, 2>
    firrtl.connect %lhs, %rhs : !firrtl.bundle<a: uint<1>, b: bundle<c: uint<1>, d flip: uint<1>>>, !firrtl.bundle<a: uint<1>, b: bundle<c: uint<1>, d flip: uint<1>>>
  }

  // CHECK-LABEL: @TestBothSidesExploded
  firrtl.module @TestBothSidesExploded() {
    // CHECK: %v1 = firrtl.wire : !firrtl.bundle<a: vector<uint<1>, 2>, b: bundle<c: vector<uint<1>, 2>, d: vector<uint<1>, 2>>>
    // CHECK: %0 = firrtl.subfield %v1[b] : !firrtl.bundle<a: vector<uint<1>, 2>, b: bundle<c: vector<uint<1>, 2>, d: vector<uint<1>, 2>>>
    // CHECK: %1 = firrtl.subfield %0[d] : !firrtl.bundle<c: vector<uint<1>, 2>, d: vector<uint<1>, 2>>
    // CHECK: %2 = firrtl.subindex %1[0] : !firrtl.vector<uint<1>, 2>
    // CHECK: %3 = firrtl.subfield %0[c] : !firrtl.bundle<c: vector<uint<1>, 2>, d: vector<uint<1>, 2>>
    // CHECK: %4 = firrtl.subindex %3[0] : !firrtl.vector<uint<1>, 2>
    // CHECK: %5 = firrtl.subfield %v1[a] : !firrtl.bundle<a: vector<uint<1>, 2>, b: bundle<c: vector<uint<1>, 2>, d: vector<uint<1>, 2>>>
    // CHECK: %6 = firrtl.subindex %5[0] : !firrtl.vector<uint<1>, 2>
    // CHECK: %v2 = firrtl.wire : !firrtl.bundle<a: vector<uint<1>, 2>, b: bundle<c: vector<uint<1>, 2>, d: vector<uint<1>, 2>>>
    // CHECK: %7 = firrtl.subfield %v2[b] : !firrtl.bundle<a: vector<uint<1>, 2>, b: bundle<c: vector<uint<1>, 2>, d: vector<uint<1>, 2>>>
    // CHECK: %8 = firrtl.subfield %7[d] : !firrtl.bundle<c: vector<uint<1>, 2>, d: vector<uint<1>, 2>>
    // CHECK: %9 = firrtl.subindex %8[0] : !firrtl.vector<uint<1>, 2>
    // CHECK: %10 = firrtl.subfield %7[c] : !firrtl.bundle<c: vector<uint<1>, 2>, d: vector<uint<1>, 2>>
    // CHECK: %11 = firrtl.subindex %10[0] : !firrtl.vector<uint<1>, 2>
    // CHECK: %12 = firrtl.subfield %v2[a] : !firrtl.bundle<a: vector<uint<1>, 2>, b: bundle<c: vector<uint<1>, 2>, d: vector<uint<1>, 2>>>
    // CHECK: %13 = firrtl.subindex %12[0] : !firrtl.vector<uint<1>, 2>
    // CHECK: firrtl.strictconnect %6, %13 : !firrtl.uint<1>
    // CHECK: firrtl.strictconnect %4, %11 : !firrtl.uint<1>
    // CHECK: firrtl.strictconnect %2, %9 : !firrtl.uint<1>
    %v1 = firrtl.wire : !firrtl.vector<bundle<a: uint<1>, b: bundle<c: uint<1>, d: uint<1>>>, 2>
    %v2 = firrtl.wire : !firrtl.vector<bundle<a: uint<1>, b: bundle<c: uint<1>, d: uint<1>>>, 2>
    %b2 = firrtl.subindex %v1[0] : !firrtl.vector<bundle<a: uint<1>, b: bundle<c: uint<1>, d: uint<1>>>, 2>
    %b3 = firrtl.subindex %v2[0] : !firrtl.vector<bundle<a: uint<1>, b: bundle<c: uint<1>, d: uint<1>>>, 2>
    firrtl.strictconnect %b2, %b3 : !firrtl.bundle<a: uint<1>, b: bundle<c: uint<1>, d: uint<1>>>
  }

  // firrtl.module @TestExplodedNode 
  firrtl.module @TestExplodedNode() {
    // CHECK: %storage = firrtl.wire : !firrtl.bundle<a: vector<uint<1>, 2>, b: vector<uint<1>, 2>>
    // CHECK: %0 = firrtl.subfield %storage[b] : !firrtl.bundle<a: vector<uint<1>, 2>, b: vector<uint<1>, 2>>
    // CHECK: %1 = firrtl.subindex %0[0] : !firrtl.vector<uint<1>, 2>
    // CHECK: %2 = firrtl.subfield %storage[a] : !firrtl.bundle<a: vector<uint<1>, 2>, b: vector<uint<1>, 2>>
    // CHECK: %3 = firrtl.subindex %2[0] : !firrtl.vector<uint<1>, 2>
    // CHECK: %4 = firrtl.bundlecreate %3, %1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.bundle<a: uint<1>, b: uint<1>>
    // CHECK: %node = firrtl.node %4 : !firrtl.bundle<a: uint<1>, b: uint<1>>    
    %storage = firrtl.wire : !firrtl.vector<bundle<a: uint<1>, b: uint<1>>, 2>
    %bundle = firrtl.subindex %storage[0] : !firrtl.vector<bundle<a: uint<1>, b: uint<1>>, 2>
    %node = firrtl.node %bundle : !firrtl.bundle<a: uint<1>, b: uint<1>>
  }

  //===--------------------------------------------------------------------===//
  // When Tests
  //===--------------------------------------------------------------------===//
  
  /// CHECK-LABEL: @TestWhen()
  firrtl.module @TestWhen() {
    // CHECK: %w = firrtl.wire : !firrtl.bundle<a: uint<8>>
    // CHECK: %0 = firrtl.subfield %w[a] : !firrtl.bundle<a: uint<8>>
    // CHECK: %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    // CHECK: firrtl.when %c1_ui1 : !firrtl.uint<1> {
    // CHECK:   %n2 = firrtl.node %0 : !firrtl.uint<8>
    // CHECK: }
    // CHECK: %n3 = firrtl.node %0 : !firrtl.uint<8>
    // CHECK: firrtl.when %c1_ui1 : !firrtl.uint<1> {
    // CHECK:   %w2 = firrtl.wire : !firrtl.bundle<a: vector<uint<8>, 2>>
    // CHECK: }
    %w = firrtl.wire : !firrtl.bundle<a: uint<8>>
    %a = firrtl.subfield %w[a] : !firrtl.bundle<a: uint<8>>
    %p = firrtl.constant 1 : !firrtl.uint<1>
    firrtl.when %p : !firrtl.uint<1> {
      %n2 = firrtl.node %a : !firrtl.uint<8>
    }
    %n3 = firrtl.node %a : !firrtl.uint<8>
    firrtl.when %p : !firrtl.uint<1> {
      %w2 = firrtl.wire : !firrtl.vector<bundle<a: uint<8>>, 2>
    }
  }

  // CHECK-LABEL: @TestWhenWithSubaccess
  firrtl.module @TestWhenWithSubaccess() {
    // CHECK: %0 = firrtl.aggregateconstant [123] : !firrtl.vector<uint<8>, 1>
    // CHECK: %c0_ui8 = firrtl.constant 0 : !firrtl.uint<8>
    // CHECK: %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    // CHECK: firrtl.when %c1_ui1 : !firrtl.uint<1> {
    // CHECK:   %2 = firrtl.subaccess %0[%c0_ui8] : !firrtl.vector<uint<8>, 1>, !firrtl.uint<8>
    // CHECK:   %nod_0 = firrtl.node %2 {name = "nod"} : !firrtl.uint<8>
    // CHECK: }
    // CHECK: %1 = firrtl.subaccess %0[%c0_ui8] : !firrtl.vector<uint<8>, 1>, !firrtl.uint<8>
    // CHECK: %nod = firrtl.node %1 : !firrtl.uint<8>
    %vec = firrtl.aggregateconstant [123] :  !firrtl.vector<uint<8>, 1>
    %idx = firrtl.constant 0 : !firrtl.uint<8>
    %cnd = firrtl.constant 1 : !firrtl.uint<1>
    firrtl.when %cnd : !firrtl.uint<1> {
      %val = firrtl.subaccess %vec[%idx] : !firrtl.vector<uint<8>, 1>, !firrtl.uint<8>
      %nod = firrtl.node %val : !firrtl.uint<8>
    }
    %val = firrtl.subaccess %vec[%idx] : !firrtl.vector<uint<8>, 1>, !firrtl.uint<8>
    %nod = firrtl.node %val : !firrtl.uint<8>
  }

  //===--------------------------------------------------------------------===//
  // Misc Tests
  //===--------------------------------------------------------------------===//

  // CHECK-LABEL: @TestDoubleSlicing()
  firrtl.module @TestDoubleSlicing() {
    // CHECK: %w = firrtl.wire : !firrtl.bundle<a: vector<vector<uint<8>, 2>, 4>, v: vector<vector<vector<uint<7>, 3>, 2>, 4>>
    %w = firrtl.wire : !firrtl.vector<vector<bundle<a: uint<8>, v: vector<uint<7>, 3>>, 2>, 4>

    // CHECK: %0 = firrtl.subfield %w[v] : !firrtl.bundle<a: vector<vector<uint<8>, 2>, 4>, v: vector<vector<vector<uint<7>, 3>, 2>, 4>>
    // CHECK: %1 = firrtl.subindex %0[0] : !firrtl.vector<vector<vector<uint<7>, 3>, 2>, 4>
    // CHECK: %2 = firrtl.subindex %1[0] : !firrtl.vector<vector<uint<7>, 3>, 2>
    // CHECK: %3 = firrtl.subindex %2[2] : !firrtl.vector<uint<7>, 3>
    // CHECK: %4 = firrtl.subfield %w[a] : !firrtl.bundle<a: vector<vector<uint<8>, 2>, 4>, v: vector<vector<vector<uint<7>, 3>, 2>, 4>>
    // CHECK: %5 = firrtl.subindex %4[0] : !firrtl.vector<vector<uint<8>, 2>, 4>
    // CHECK: %6 = firrtl.subindex %5[0] : !firrtl.vector<uint<8>, 2>

    // CHECK: %7 = firrtl.bundlecreate %5, %1 : (!firrtl.vector<uint<8>, 2>, !firrtl.vector<vector<uint<7>, 3>, 2>) -> !firrtl.bundle<a: vector<uint<8>, 2>, v: vector<vector<uint<7>, 3>, 2>>
    // CHECK: %n_0 = firrtl.node %7 : !firrtl.bundle<a: vector<uint<8>, 2>, v: vector<vector<uint<7>, 3>, 2>>
    %w_0 = firrtl.subindex %w[0] : !firrtl.vector<vector<bundle<a: uint<8>, v: vector<uint<7>, 3>>, 2>, 4>
    %n_0 = firrtl.node %w_0 : !firrtl.vector<bundle<a: uint<8>, v: vector<uint<7>, 3>>, 2>

    // CHECK: %8 = firrtl.bundlecreate %6, %2 : (!firrtl.uint<8>, !firrtl.vector<uint<7>, 3>) -> !firrtl.bundle<a: uint<8>, v: vector<uint<7>, 3>>
    // CHECK: %n_0_1 = firrtl.node %8 : !firrtl.bundle<a: uint<8>, v: vector<uint<7>, 3>>
    %w_0_1 = firrtl.subindex %w_0[0] : !firrtl.vector<bundle<a: uint<8>, v: vector<uint<7>, 3>>, 2>
    %n_0_1 = firrtl.node %w_0_1 : !firrtl.bundle<a: uint<8>, v: vector<uint<7>, 3>>

    // CHECK: %n_0_1_b = firrtl.node %2 : !firrtl.vector<uint<7>, 3>
    %w_0_1_b = firrtl.subfield %w_0_1[v] : !firrtl.bundle<a: uint<8>, v: vector<uint<7>, 3>>
    %n_0_1_b = firrtl.node %w_0_1_b : !firrtl.vector<uint<7>, 3>
  
    // CHECK: %n_0_1_b_2 = firrtl.node %3 : !firrtl.uint<7>
    %w_0_1_b_2 = firrtl.subindex %w_0_1_b[2] : !firrtl.vector<uint<7>, 3>
    %n_0_1_b_2 = firrtl.node %w_0_1_b_2 : !firrtl.uint<7>
  }

  // connect with flip, rhs is a rematerialized bundle, Do we preserve the flip?
  // CHECK-LABEL: @VBF
  firrtl.module @VBF(
    // CHECK-SAME: in %i: !firrtl.bundle<a: vector<uint<8>, 2>, b flip: vector<uint<8>, 2>>
    in  %i : !firrtl.vector<bundle<a: uint<8>, b flip: uint<8>>, 2>,
    // CHECK-SAME: out %o: !firrtl.bundle<a: uint<8>, b flip: uint<8>>
    out %o : !firrtl.bundle<a: uint<8>, b flip: uint<8>>) {
    // CHECK: %0 = firrtl.subfield %o[b] : !firrtl.bundle<a: uint<8>, b flip: uint<8>>
    // CHECK: %1 = firrtl.subfield %o[a] : !firrtl.bundle<a: uint<8>, b flip: uint<8>>
    // CHECK: %2 = firrtl.subfield %i[b] : !firrtl.bundle<a: vector<uint<8>, 2>, b flip: vector<uint<8>, 2>>
    // CHECK: %3 = firrtl.subindex %2[0] : !firrtl.vector<uint<8>, 2>
    // CHECK: %4 = firrtl.subfield %i[a] : !firrtl.bundle<a: vector<uint<8>, 2>, b flip: vector<uint<8>, 2>>
    // CHECK: %5 = firrtl.subindex %4[0] : !firrtl.vector<uint<8>, 2>
    // CHECK: firrtl.connect %1, %5 : !firrtl.uint<8>
    // CHECK: firrtl.connect %3, %0 : !firrtl.uint<8>
    %0 = firrtl.subindex %i[0] : !firrtl.vector<bundle<a: uint<8>, b flip: uint<8>>, 2>
    firrtl.connect %o, %0 : !firrtl.bundle<a: uint<8>, b flip: uint<8>>, !firrtl.bundle<a: uint<8>, b flip: uint<8>>
  }

  // connect lhs is an exploded bundle with flip, Do we connect in the right direction?
  // CHECK-LABEL: VBF2
  firrtl.module @VBF2(
    // CHECK-SAME: in %i: !firrtl.bundle<a: uint<8>, b flip: uint<8>>
    in  %i : !firrtl.bundle<a: uint<8>, b flip: uint<8>>,
    // CHECK-SAME: out %o: !firrtl.bundle<a: vector<uint<8>, 2>, b flip: vector<uint<8>, 2>>
    out %o : !firrtl.vector<bundle<a: uint<8>, b flip: uint<8>>, 2>) {
    // CHECK: %0 = firrtl.subfield %i[b] : !firrtl.bundle<a: uint<8>, b flip: uint<8>>
    // CHECK: %1 = firrtl.subfield %i[a] : !firrtl.bundle<a: uint<8>, b flip: uint<8>>
    // CHECK: %2 = firrtl.subfield %o[b] : !firrtl.bundle<a: vector<uint<8>, 2>, b flip: vector<uint<8>, 2>>
    // CHECK: %3 = firrtl.subindex %2[0] : !firrtl.vector<uint<8>, 2>
    // CHECK: %4 = firrtl.subfield %o[a] : !firrtl.bundle<a: vector<uint<8>, 2>, b flip: vector<uint<8>, 2>>
    // CHECK: %5 = firrtl.subindex %4[0] : !firrtl.vector<uint<8>, 2>
    %0 = firrtl.subindex %o[0] : !firrtl.vector<bundle<a: uint<8>, b flip: uint<8>>, 2>
    // CHECK: firrtl.connect %5, %1 : !firrtl.uint<8>
    // CHECK: firrtl.connect %0, %3 : !firrtl.uint<8>
    firrtl.connect %0, %i : !firrtl.bundle<a: uint<8>, b flip: uint<8>>, !firrtl.bundle<a: uint<8>, b flip: uint<8>>
  }

  // CHECK-LABEL: TestBundleCreate_VB
  firrtl.module @TestBundleCreate_VB(out %out : !firrtl.vector<bundle<a: uint<8>, b: uint<16>>, 2>) {
    // CHECK: %c1_ui8 = firrtl.constant 1 : !firrtl.uint<8>
    %0 = firrtl.constant 1 : !firrtl.uint<8>
  
    // CHECK: %c2_ui16 = firrtl.constant 2 : !firrtl.uint<16>
    %1 = firrtl.constant 2 : !firrtl.uint<16>
  
    // CHECK: %0 = firrtl.bundlecreate %c1_ui8, %c2_ui16 : (!firrtl.uint<8>, !firrtl.uint<16>) -> !firrtl.bundle<a: uint<8>, b: uint<16>>
    // CHECK: %1 = firrtl.subfield %0[b] : !firrtl.bundle<a: uint<8>, b: uint<16>>
    // CHECK: %2 = firrtl.subfield %0[a] : !firrtl.bundle<a: uint<8>, b: uint<16>>
    %bundle1 = firrtl.bundlecreate %0, %1 : (!firrtl.uint<8>, !firrtl.uint<16>) -> !firrtl.bundle<a: uint<8>, b: uint<16>>

    // CHECK: %3 = firrtl.bundlecreate %c1_ui8, %c2_ui16 : (!firrtl.uint<8>, !firrtl.uint<16>) -> !firrtl.bundle<a: uint<8>, b: uint<16>>
    // CHECK: %4 = firrtl.subfield %3[b] : !firrtl.bundle<a: uint<8>, b: uint<16>>
    // CHECK: %5 = firrtl.subfield %3[a] : !firrtl.bundle<a: uint<8>, b: uint<16>>
    %bundle2 = firrtl.bundlecreate %0, %1 : (!firrtl.uint<8>, !firrtl.uint<16>) -> !firrtl.bundle<a: uint<8>, b: uint<16>>
    
    // CHECK: %6 = firrtl.vectorcreate %2, %5 : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.vector<uint<8>, 2>
    // CHECK: %7 = firrtl.vectorcreate %1, %4 : (!firrtl.uint<16>, !firrtl.uint<16>) -> !firrtl.vector<uint<16>, 2>
    // CHECK: %8 = firrtl.bundlecreate %6, %7 : (!firrtl.vector<uint<8>, 2>, !firrtl.vector<uint<16>, 2>) -> !firrtl.bundle<a: vector<uint<8>, 2>, b: vector<uint<16>, 2>>
    %vector  = firrtl.vectorcreate %bundle1, %bundle2 : (!firrtl.bundle<a: uint<8>, b: uint<16>>, !firrtl.bundle<a: uint<8>, b: uint<16>>) -> !firrtl.vector<bundle<a: uint<8>, b: uint<16>>, 2>
    
    // CHECK: firrtl.connect %out, %8 : !firrtl.bundle<a: vector<uint<8>, 2>, b: vector<uint<16>, 2>>, !firrtl.bundle<a: vector<uint<8>, 2>, b: vector<uint<16>, 2>>
    firrtl.connect %out, %vector : !firrtl.vector<bundle<a: uint<8>, b: uint<16>>, 2>, !firrtl.vector<bundle<a: uint<8>, b: uint<16>>, 2>
  }

  //===--------------------------------------------------------------------===//
  // Ref Type Tests
  //===--------------------------------------------------------------------===//

  // CHECK-LABEL: @RefSender
  firrtl.module @RefSender(out %port: !firrtl.probe<vector<bundle<a: uint<4>, b: uint<8>>, 2>>) {
   // CHECK: %w = firrtl.wire : !firrtl.bundle<a: vector<uint<4>, 2>, b: vector<uint<8>, 2>>
    // CHECK: %0 = firrtl.ref.send %w : !firrtl.bundle<a: vector<uint<4>, 2>, b: vector<uint<8>, 2>>
    // CHECK: firrtl.ref.define %port, %0 : !firrtl.probe<bundle<a: vector<uint<4>, 2>, b: vector<uint<8>, 2>>>
    %w = firrtl.wire : !firrtl.vector<bundle<a: uint<4>, b: uint<8>>, 2>
    %ref = firrtl.ref.send %w : !firrtl.vector<bundle<a: uint<4>, b: uint<8>>, 2>
    firrtl.ref.define %port, %ref : !firrtl.probe<vector<bundle<a: uint<4>, b: uint<8>>, 2>>
  }

  firrtl.module @RefResolver() {
    // CHECK: %sender_port = firrtl.instance sender @RefSender(out port: !firrtl.probe<bundle<a: vector<uint<4>, 2>, b: vector<uint<8>, 2>>>)
    // CHECK: %0 = firrtl.ref.sub %sender_port[1] : !firrtl.probe<bundle<a: vector<uint<4>, 2>, b: vector<uint<8>, 2>>>
    // CHECK: %1 = firrtl.ref.sub %0[1] : !firrtl.probe<vector<uint<8>, 2>>
    // CHECK: %2 = firrtl.ref.sub %sender_port[0] : !firrtl.probe<bundle<a: vector<uint<4>, 2>, b: vector<uint<8>, 2>>>
    // CHECK: %3 = firrtl.ref.sub %2[1] : !firrtl.probe<vector<uint<4>, 2>>
    // CHECK: %4 = firrtl.ref.resolve %3 : !firrtl.probe<uint<4>>
    // CHECK: %5 = firrtl.ref.resolve %1 : !firrtl.probe<uint<8>>
    // CHECK: %6 = firrtl.bundlecreate %4, %5 : (!firrtl.uint<4>, !firrtl.uint<8>) -> !firrtl.bundle<a: uint<4>, b: uint<8>>
    // CHECK: %w = firrtl.wire : !firrtl.bundle<a: uint<4>, b: uint<8>>
    // CHECK: firrtl.strictconnect %w, %6 : !firrtl.bundle<a: uint<4>, b: uint<8>>
    %vector_ref = firrtl.instance sender @RefSender(out port: !firrtl.probe<vector<bundle<a: uint<4>, b: uint<8>>, 2>>)
    %bundle_ref = firrtl.ref.sub     %vector_ref[1] : !firrtl.probe<vector<bundle<a: uint<4>, b: uint<8>>, 2>>
    %bundle_val = firrtl.ref.resolve %bundle_ref    : !firrtl.probe<bundle<a: uint<4>, b: uint<8>>>
    %w = firrtl.wire: !firrtl.bundle<a: uint<4>, b: uint<8>>
    firrtl.strictconnect %w, %bundle_val : !firrtl.bundle<a: uint<4>, b: uint<8>>
  }

  //===--------------------------------------------------------------------===//
  // Annotation Tests
  //===--------------------------------------------------------------------===//

  // CHECK-LABEL: firrtl.module @Annotations(
  firrtl.module @Annotations(
    // CHECK-SAME: in %in: !firrtl.bundle<f: vector<uint<8>, 4>, g: vector<uint<8>, 4>>
    in %in: !firrtl.vector<bundle<f: uint<8>, g: uint<8>>, 4> [
      // CHECK-SAME: {circt.fieldID = 2 : i64, class = "f"}
      // CHECK-SAME: {circt.fieldID = 7 : i64, class = "f"}
      {class = "f", circt.fieldID = 1 : i64}
    ]
    ) {

    // CHECK: %w = firrtl.wire {
    %w = firrtl.wire {
      annotations =  [
        // CHECK-SAME: {circt.fieldID = 0 : i64, class = "0"}
        {circt.fieldID = 0 : i64, class = "0"},
        // CHECK-SAME: {circt.fieldID = 2 : i64, class = "1"}
        // CHECK-SAME: {circt.fieldID = 7 : i64, class = "1"}
        {circt.fieldID = 1 : i64, class = "1"},
        // CHECK-SAME: {circt.fieldID = 2 : i64, class = "2"}
        {circt.fieldID = 2 : i64, class = "2"},
        // CHECK-SAME: {circt.fieldID = 7 : i64, class = "3"}
        {circt.fieldID = 3 : i64, class = "3"},
        // CHECK-SAME: {circt.fieldID = 3 : i64, class = "4"}
        // CHECK-SAME: {circt.fieldID = 8 : i64, class = "4"}
        {circt.fieldID = 4 : i64, class = "4"},
        // CHECK-SAME: {circt.fieldID = 3 : i64, class = "5"}
        {circt.fieldID = 5 : i64, class = "5"},
        // CHECK-SAME: {circt.fieldID = 8 : i64, class = "6"}
        {circt.fieldID = 6 : i64, class = "6"}
    ]} : 
    // CHECK-SAME: !firrtl.bundle<a: vector<uint<8>, 4>, b: vector<uint<8>, 4>>
    !firrtl.vector<bundle<a: uint<8>, b: uint<8>>, 4>
    
    // Targeting the bundle of the data field should explode and retarget to the
    // first element of the field vector.
    // CHECK: firrtl.mem
    // CHECK-SAME{LITERAL}: portAnnotations = [[{circt.fieldID = 6 : i64, class = "mem0"}]]
    %bar_r = firrtl.mem Undefined  {depth = 16 : i64, name = "bar", portAnnotations = [[{circt.fieldID = 5 : i64, class = "mem0"}]], portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: vector<bundle<a: uint<8>>, 5>> 
  }
}
