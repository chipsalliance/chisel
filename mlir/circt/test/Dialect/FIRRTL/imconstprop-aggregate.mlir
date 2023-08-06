// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-imconstprop))' --split-input-file  %s | FileCheck %s

// This contains a lot of tests which should be caught by IMCP.
// For now, we are checking that the aggregates don't cause the pass to error out.

firrtl.circuit "VectorPropagation1" {
  // CHECK-LABEL: @VectorPropagation1
  firrtl.module @VectorPropagation1(out %b: !firrtl.uint<1>) {
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %tmp = firrtl.wire : !firrtl.vector<uint<1>, 2>
    %0 = firrtl.subindex %tmp[0] : !firrtl.vector<uint<1>, 2>
    %1 = firrtl.subindex %tmp[1] : !firrtl.vector<uint<1>, 2>
    %2 = firrtl.xor %0, %1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.strictconnect %0, %c1_ui1 : !firrtl.uint<1>
    firrtl.strictconnect %1, %c1_ui1 : !firrtl.uint<1>
    // CHECK: firrtl.strictconnect %b, %c0_ui1 : !firrtl.uint<1>
    firrtl.strictconnect %b, %2 : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "VectorPropagation2" {
  // CHECK-LABEL: @VectorPropagation2
  firrtl.module @VectorPropagation2(out %b1: !firrtl.uint<6>, out %b2: !firrtl.uint<6>, out %b3: !firrtl.uint<6>) {

    // tmp1[0][0] <= 1
    // tmp1[0][1] <= 2
    // tmp1[1][0] <= 4
    // tmp1[1][1] <= 8
    // tmp1[2][0] <= 16
    // tmp1[2][1] <= 32

    // b1 <= tmp[0][0] xor tmp1[1][0] = 5
    // b2 <= tmp[2][1] xor tmp1[0][1] = 34
    // b3 <= tmp[1][1] xor tmp1[2][0] = 24

    %c32_ui6 = firrtl.constant 32 : !firrtl.uint<6>
    %c16_ui6 = firrtl.constant 16 : !firrtl.uint<6>
    %c8_ui6 = firrtl.constant 8 : !firrtl.uint<6>
    %c4_ui6 = firrtl.constant 4 : !firrtl.uint<6>
    %c2_ui6 = firrtl.constant 2 : !firrtl.uint<6>
    %c1_ui6 = firrtl.constant 1 : !firrtl.uint<6>
    %tmp = firrtl.wire  : !firrtl.vector<vector<uint<6>, 2>, 3>
    %0 = firrtl.subindex %tmp[0] : !firrtl.vector<vector<uint<6>, 2>, 3>
    %1 = firrtl.subindex %0[0] : !firrtl.vector<uint<6>, 2>
    firrtl.strictconnect %1, %c1_ui6 : !firrtl.uint<6>
    %2 = firrtl.subindex %0[1] : !firrtl.vector<uint<6>, 2>
    firrtl.strictconnect %2, %c2_ui6 : !firrtl.uint<6>
    %3 = firrtl.subindex %tmp[1] : !firrtl.vector<vector<uint<6>, 2>, 3>
    %4 = firrtl.subindex %3[0] : !firrtl.vector<uint<6>, 2>
    firrtl.strictconnect %4, %c4_ui6 : !firrtl.uint<6>
    %5 = firrtl.subindex %3[1] : !firrtl.vector<uint<6>, 2>
    firrtl.strictconnect %5, %c8_ui6 : !firrtl.uint<6>
    %6 = firrtl.subindex %tmp[2] : !firrtl.vector<vector<uint<6>, 2>, 3>
    %7 = firrtl.subindex %6[0] : !firrtl.vector<uint<6>, 2>
    firrtl.strictconnect %7, %c16_ui6 : !firrtl.uint<6>
    %8 = firrtl.subindex %6[1] : !firrtl.vector<uint<6>, 2>
    firrtl.strictconnect %8, %c32_ui6 : !firrtl.uint<6>
    %9 = firrtl.xor %1, %4 : (!firrtl.uint<6>, !firrtl.uint<6>) -> !firrtl.uint<6>
    firrtl.strictconnect %b1, %9 : !firrtl.uint<6>
    %10 = firrtl.xor %8, %2 : (!firrtl.uint<6>, !firrtl.uint<6>) -> !firrtl.uint<6>
    firrtl.strictconnect %b2, %10 : !firrtl.uint<6>
    %11 = firrtl.xor %7, %5 : (!firrtl.uint<6>, !firrtl.uint<6>) -> !firrtl.uint<6>
    firrtl.strictconnect %b3, %11 : !firrtl.uint<6>
    // CHECK:      firrtl.strictconnect %b1, %c5_ui6 : !firrtl.uint<6>
    // CHECK-NEXT: firrtl.strictconnect %b2, %c34_ui6 : !firrtl.uint<6>
    // CHECK-NEXT: firrtl.strictconnect %b3, %c24_ui6 : !firrtl.uint<6>
  }
}

// -----

firrtl.circuit "BundlePropagation1"   {
  // CHECK-LABEL: @BundlePropagation1
  firrtl.module @BundlePropagation1(out %result: !firrtl.uint<3>) {
    %tmp = firrtl.wire  : !firrtl.bundle<a: uint<3>, b: uint<3>, c: uint<3>>
    %c1_ui3 = firrtl.constant 1 : !firrtl.uint<3>
    %c2_ui3 = firrtl.constant 2 : !firrtl.uint<3>
    %c4_ui3 = firrtl.constant 4 : !firrtl.uint<3>
    %0 = firrtl.subfield %tmp[a] : !firrtl.bundle<a: uint<3>, b: uint<3>, c: uint<3>>
    %1 = firrtl.subfield %tmp[b] : !firrtl.bundle<a: uint<3>, b: uint<3>, c: uint<3>>
    %2 = firrtl.subfield %tmp[c] : !firrtl.bundle<a: uint<3>, b: uint<3>, c: uint<3>>
    firrtl.strictconnect %0, %c1_ui3 : !firrtl.uint<3>
    firrtl.strictconnect %1, %c2_ui3 : !firrtl.uint<3>
    firrtl.strictconnect %2, %c4_ui3 : !firrtl.uint<3>
    %3 = firrtl.xor %0, %1 : (!firrtl.uint<3>, !firrtl.uint<3>) -> !firrtl.uint<3>
    %4 = firrtl.xor %3, %2 : (!firrtl.uint<3>, !firrtl.uint<3>) -> !firrtl.uint<3>
    firrtl.strictconnect %result, %4 : !firrtl.uint<3>
    // CHECK:  firrtl.strictconnect %result, %c7_ui3 : !firrtl.uint<3>
  }
}

// -----

firrtl.circuit "DontTouchAggregate" {
  firrtl.module @DontTouchAggregate(in %clock: !firrtl.clock, out %out1: !firrtl.uint<1>, out %out2: !firrtl.uint<1>) {
    %init = firrtl.wire sym @dntSym: !firrtl.vector<uint<1>, 2>
    %0 = firrtl.subindex %init[0] : !firrtl.vector<uint<1>, 2>
    %1 = firrtl.subindex %init[1] : !firrtl.vector<uint<1>, 2>
    %true = firrtl.constant 1 : !firrtl.uint<1>
    firrtl.strictconnect %0, %true : !firrtl.uint<1>
    firrtl.strictconnect %1, %true : !firrtl.uint<1>

    // CHECK:      firrtl.strictconnect %out1, %0 : !firrtl.uint<1>
    // CHECK-NEXT: firrtl.strictconnect %out2, %1 : !firrtl.uint<1>
    firrtl.strictconnect %out1, %0 : !firrtl.uint<1>
    firrtl.strictconnect %out2, %1 : !firrtl.uint<1>
  }
}

// -----
// Following tests are ported from normal imconstprop tests.

firrtl.circuit "OutPortTop" {
  // Check that we don't propagate througth it.
  firrtl.module @OutPortChild(out %out: !firrtl.vector<uint<1>, 2> sym @dntSym)
  {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %0 = firrtl.subindex %out[0] : !firrtl.vector<uint<1>, 2>
    %1 = firrtl.subindex %out[1] : !firrtl.vector<uint<1>, 2>
    firrtl.strictconnect %0, %c0_ui1 : !firrtl.uint<1>
    firrtl.strictconnect %1, %c0_ui1 : !firrtl.uint<1>
  }
  // CHECK-LABEL: firrtl.module @OutPortTop
  firrtl.module @OutPortTop(out %out1: !firrtl.uint<1>, out %out2: !firrtl.uint<1>) {
    %c_out = firrtl.instance c @OutPortChild(out out: !firrtl.vector<uint<1>, 2>)
    %0 = firrtl.subindex %c_out[0] : !firrtl.vector<uint<1>, 2>
    %1 = firrtl.subindex %c_out[1] : !firrtl.vector<uint<1>, 2>
    firrtl.strictconnect %out1, %0 : !firrtl.uint<1>
    firrtl.strictconnect %out2, %1 : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "InputPortTop"  {
  // CHECK-LABEL: firrtl.module private @InputPortChild2
  firrtl.module private @InputPortChild2(in %in0: !firrtl.bundle<v: uint<1>>, in %in1: !firrtl.bundle<v: uint<1>>, out %out: !firrtl.bundle<v: uint<1>>) {
    // CHECK: firrtl.and %0, %c1_ui1
    %0 = firrtl.subfield %in1[v] : !firrtl.bundle<v: uint<1>>
    %1 = firrtl.subfield %in0[v] : !firrtl.bundle<v: uint<1>>
    %2 = firrtl.subfield %out[v] : !firrtl.bundle<v: uint<1>>
    %3 = firrtl.and %1, %0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.strictconnect %2, %3 : !firrtl.uint<1>
  }
  // CHECK-LABEL: firrtl.module private @InputPortChild
  firrtl.module private @InputPortChild(in %in0: !firrtl.bundle<v: uint<1>>,
    in %in1: !firrtl.bundle<v: uint<1>> sym @dntSym,
    out %out: !firrtl.bundle<v: uint<1>>)
  {
    // CHECK: firrtl.and %1, %0
    %0 = firrtl.subfield %in1[v] : !firrtl.bundle<v: uint<1>>
    %1 = firrtl.subfield %in0[v] : !firrtl.bundle<v: uint<1>>
    %2 = firrtl.subfield %out[v] : !firrtl.bundle<v: uint<1>>
    %3 = firrtl.and %1, %0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.strictconnect %2, %3 : !firrtl.uint<1>
  }

  // CHECK-LABEL: firrtl.module @InputPortTop
  firrtl.module @InputPortTop(in %x: !firrtl.bundle<v: uint<1>>, out %z: !firrtl.bundle<v: uint<1>>, out %z2: !firrtl.bundle<v: uint<1>>) {
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %0 = firrtl.subfield %z2[v] : !firrtl.bundle<v: uint<1>>
    %1 = firrtl.subfield %x[v] : !firrtl.bundle<v: uint<1>>
    %2 = firrtl.subfield %z[v] : !firrtl.bundle<v: uint<1>>
    %c_in0, %c_in1, %c_out = firrtl.instance c  @InputPortChild(in in0: !firrtl.bundle<v: uint<1>>, in in1: !firrtl.bundle<v: uint<1>>, out out: !firrtl.bundle<v: uint<1>>)
    %3 = firrtl.subfield %c_in1[v] : !firrtl.bundle<v: uint<1>>
    %4 = firrtl.subfield %c_in0[v] : !firrtl.bundle<v: uint<1>>
    %5 = firrtl.subfield %c_out[v] : !firrtl.bundle<v: uint<1>>
    %c2_in0, %c2_in1, %c2_out = firrtl.instance c2  @InputPortChild2(in in0: !firrtl.bundle<v: uint<1>>, in in1: !firrtl.bundle<v: uint<1>>, out out: !firrtl.bundle<v: uint<1>>)
    %6 = firrtl.subfield %c2_in1[v] : !firrtl.bundle<v: uint<1>>
    %7 = firrtl.subfield %c2_in0[v] : !firrtl.bundle<v: uint<1>>
    %8 = firrtl.subfield %c2_out[v] : !firrtl.bundle<v: uint<1>>
    firrtl.strictconnect %2, %5 : !firrtl.uint<1>
    firrtl.strictconnect %4, %1 : !firrtl.uint<1>
    firrtl.strictconnect %3, %c1_ui1 : !firrtl.uint<1>
    firrtl.strictconnect %0, %8 : !firrtl.uint<1>
    firrtl.strictconnect %7, %1 : !firrtl.uint<1>
    firrtl.strictconnect %6, %c1_ui1 : !firrtl.uint<1>
  }
}

// -----

// CHECK-LABEL: firrtl.circuit "rhs_sink_output_used_as_wire"
// This test checks that an output port sink, used as a RHS of a connect, is not
// optimized away.  This is similar to the oscillator tests above, but more
// reduced. See:
//   - https://github.com/llvm/circt/issues/1488
//
firrtl.circuit "rhs_sink_output_used_as_wire" {
  // CHECK-LABEL: firrtl.module private @Bar
  firrtl.module private @Bar(in %a: !firrtl.bundle<v: uint<1>>, in %b: !firrtl.bundle<v: uint<1>>, out %c: !firrtl.bundle<v: uint<1>>, out %d: !firrtl.bundle<v: uint<1>>) {
    %0 = firrtl.subfield %d[v] : !firrtl.bundle<v: uint<1>>
    %1 = firrtl.subfield %a[v] : !firrtl.bundle<v: uint<1>>
    %2 = firrtl.subfield %b[v] : !firrtl.bundle<v: uint<1>>
    %3 = firrtl.subfield %c[v] : !firrtl.bundle<v: uint<1>>
    firrtl.strictconnect %3, %2 : !firrtl.uint<1>
    %_c = firrtl.wire  : !firrtl.bundle<v: uint<1>>
    %4 = firrtl.subfield %_c[v] : !firrtl.bundle<v: uint<1>>
    %5 = firrtl.xor %1, %3 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.strictconnect %4, %5 : !firrtl.uint<1>
    firrtl.strictconnect %0, %4 : !firrtl.uint<1>
  }

  // CHECK-LABEL: firrtl.module @rhs_sink_output_used_as_wire
  firrtl.module @rhs_sink_output_used_as_wire(in %a: !firrtl.bundle<v: uint<1>>, in %b: !firrtl.bundle<v: uint<1>>, out %c: !firrtl.bundle<v: uint<1>>, out %d: !firrtl.bundle<v: uint<1>>) {
    %bar_a, %bar_b, %bar_c, %bar_d = firrtl.instance bar  @Bar(in a: !firrtl.bundle<v: uint<1>>, in b: !firrtl.bundle<v: uint<1>>, out c: !firrtl.bundle<v: uint<1>>, out d: !firrtl.bundle<v: uint<1>>)
    %0 = firrtl.subfield %a[v] : !firrtl.bundle<v: uint<1>>
    %1 = firrtl.subfield %bar_a[v] : !firrtl.bundle<v: uint<1>>
    firrtl.strictconnect %1, %0 : !firrtl.uint<1>
    %2 = firrtl.subfield %b[v] : !firrtl.bundle<v: uint<1>>
    %3 = firrtl.subfield %bar_b[v] : !firrtl.bundle<v: uint<1>>
    firrtl.strictconnect %3, %2 : !firrtl.uint<1>
    %4 = firrtl.subfield %bar_c[v] : !firrtl.bundle<v: uint<1>>
    %5 = firrtl.subfield %c[v] : !firrtl.bundle<v: uint<1>>
    firrtl.strictconnect %5, %4 : !firrtl.uint<1>
    %6 = firrtl.subfield %bar_d[v] : !firrtl.bundle<v: uint<1>>
    %7 = firrtl.subfield %d[v] : !firrtl.bundle<v: uint<1>>
    firrtl.strictconnect %7, %6 : !firrtl.uint<1>
  }
}

// -----
firrtl.circuit "dntOutput"  {
  // CHECK-LABEL: firrtl.module @dntOutput
  // CHECK:      %[[INT_B_V:.+]] = firrtl.subfield %int_b[v] : !firrtl.bundle<v: uint<3>>
  // CHECK-NEXT: %[[MUX:.+]] = firrtl.mux(%c, %[[INT_B_V]], %c2_ui3)
  firrtl.module @dntOutput(out %b: !firrtl.bundle<v: uint<3>>, in %c: !firrtl.uint<1>) {
    %c2_ui3 = firrtl.constant 2 : !firrtl.uint<3>
    %0 = firrtl.subfield %b[v] : !firrtl.bundle<v: uint<3>>
    %int_b = firrtl.instance int  @foo(out b: !firrtl.bundle<v: uint<3>>)
    %1 = firrtl.subfield %int_b[v] : !firrtl.bundle<v: uint<3>>
    %2 = firrtl.mux(%c, %1, %c2_ui3) : (!firrtl.uint<1>, !firrtl.uint<3>, !firrtl.uint<3>) -> !firrtl.uint<3>
    firrtl.strictconnect %0, %2 : !firrtl.uint<3>
  }
  firrtl.module private @foo(out %b: !firrtl.bundle<v: uint<3>> sym @dntSym1){
    %c1_ui3 = firrtl.constant 1 : !firrtl.uint<3>
    %0 = firrtl.subfield %b[v] : !firrtl.bundle<v: uint<3>>
    firrtl.strictconnect %0, %c1_ui3 : !firrtl.uint<3>
  }
}

// -----

firrtl.circuit "Issue4369"  {
  // CHECK-LABEL: firrtl.module private @Bar
  firrtl.module private @Bar(in %in: !firrtl.vector<uint<1>, 1>, out %out: !firrtl.uint<1>) {
    %0 = firrtl.subindex %in[0] : !firrtl.vector<uint<1>, 1>
    %a = firrtl.wire   : !firrtl.uint<1>
    // CHECK: firrtl.strictconnect %a, %0
    // CHECK-NEXT: firrtl.strictconnect %out, %a
    firrtl.strictconnect %a, %0 : !firrtl.uint<1>
    firrtl.strictconnect %out, %a : !firrtl.uint<1>
  }
  firrtl.module @Issue4369(in %a_0: !firrtl.uint<1>, out %b: !firrtl.uint<1>) {
    %bar_in, %bar_out = firrtl.instance bar  @Bar(in in: !firrtl.vector<uint<1>, 1>, out out: !firrtl.uint<1>)
    %0 = firrtl.subindex %bar_in[0] : !firrtl.vector<uint<1>, 1>
    firrtl.strictconnect %0, %a_0 : !firrtl.uint<1>
    firrtl.strictconnect %b, %bar_out : !firrtl.uint<1>
  }
}

// -----
firrtl.circuit "AggregateConstant"  {
  // CHECK-LABEL: AggregateConstant
  firrtl.module @AggregateConstant(out %out: !firrtl.uint<1>) {
    %0 = firrtl.aggregateconstant [0 : ui1, 1 : ui1] : !firrtl.vector<uint<1>, 2>
    %w = firrtl.wire : !firrtl.vector<uint<1>, 2>
    %1 = firrtl.subindex %w[1] : !firrtl.vector<uint<1>, 2>
    firrtl.strictconnect %out, %1 : !firrtl.uint<1>
    // CHECK: firrtl.strictconnect %out, %c1_ui1
    firrtl.strictconnect %w, %0 : !firrtl.vector<uint<1>, 2>
  }
}
