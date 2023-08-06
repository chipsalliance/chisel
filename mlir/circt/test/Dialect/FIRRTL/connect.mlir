
// RUN: circt-opt %s | FileCheck %s

firrtl.circuit "reset0" {

// Reset destination.
firrtl.module @reset0(in %a : !firrtl.uint<1>, out %b : !firrtl.reset) {
  // CHECK: firrtl.connect %b, %a
  firrtl.connect %b, %a : !firrtl.reset, !firrtl.uint<1>
}

firrtl.module @reset1(in %a : !firrtl.asyncreset, out %b : !firrtl.reset) {
  // CHECK: firrtl.connect %b, %a
  firrtl.connect %b, %a : !firrtl.reset, !firrtl.asyncreset
}

/// Reset types can be connected to Reset, UInt<1>, or AsyncReset types.

// Reset source.
firrtl.module @reset2(in %a : !firrtl.reset, out %b : !firrtl.reset) {
  // CHECK: firrtl.connect %b, %a
  firrtl.connect %b, %a : !firrtl.reset, !firrtl.reset
}

firrtl.module @reset3(in %a : !firrtl.reset, out %b : !firrtl.uint<1>) {
  // CHECK: firrtl.connect %b, %a
  firrtl.connect %b, %a : !firrtl.uint<1>, !firrtl.reset
}

firrtl.module @reset4(in %a : !firrtl.reset, out %b : !firrtl.asyncreset) {
  // CHECK: firrtl.connect %b, %a
  firrtl.connect %b, %a : !firrtl.asyncreset, !firrtl.reset
}

// AsyncReset source.
firrtl.module @asyncreset0(in %a : !firrtl.asyncreset, out %b : !firrtl.asyncreset) {
  // CHECK: firrtl.connect %b, %a
  firrtl.connect %b, %a : !firrtl.asyncreset, !firrtl.asyncreset
}

// Clock source.
firrtl.module @clock0(in %a : !firrtl.clock, out %b : !firrtl.clock) {
  // CHECK: firrtl.connect %b, %a
  firrtl.connect %b, %a : !firrtl.clock, !firrtl.clock
}

/// Ground types can be connected if they are the same ground type.

// SInt<> source.
firrtl.module @sint0(in %a : !firrtl.sint<1>, out %b : !firrtl.sint<1>) {
  // CHECK: firrtl.connect %b, %a
  firrtl.connect %b, %a : !firrtl.sint<1>, !firrtl.sint<1>
}

// UInt<> source.
firrtl.module @uint0(in %a : !firrtl.uint<1>, out %b : !firrtl.uint<1>) {
  // CHECK: firrtl.connect %b, %a
  firrtl.connect %b, %a : !firrtl.uint<1>, !firrtl.uint<1>
}
firrtl.module @uint1(in %a : !firrtl.uint<1>, out %b : !firrtl.uint<2>) {
  // CHECK: firrtl.connect %b, %a
  firrtl.connect %b, %a : !firrtl.uint<2>, !firrtl.uint<1>
}

/// Vector types can be connected if they have the same size and element type.
firrtl.module @vect0(in %a : !firrtl.vector<uint<1>, 3>, out %b : !firrtl.vector<uint<1>, 3>) {
  // CHECK: firrtl.connect %b, %a
  firrtl.connect %b, %a : !firrtl.vector<uint<1>, 3>, !firrtl.vector<uint<1>, 3>
}

/// Bundle types can be connected if they have the same size, element names, and
/// element types.

firrtl.module @bundle0(in %a : !firrtl.bundle<f1: uint<1>, f2 flip: sint<1>>, out %b : !firrtl.bundle<f1: uint<1>, f2 flip: sint<1>>) {
  // CHECK: firrtl.connect %b, %a
  firrtl.connect %b, %a : !firrtl.bundle<f1: uint<1>, f2 flip: sint<1>>, !firrtl.bundle<f1: uint<1>, f2 flip: sint<1>>
}

firrtl.module @bundle1(in %a : !firrtl.bundle<f1: uint<1>, f2 flip: sint<2>>, out %b : !firrtl.bundle<f1: uint<2>, f2 flip: sint<1>>) {
  // CHECK: firrtl.connect %b, %a
  firrtl.connect %b, %a : !firrtl.bundle<f1: uint<2>, f2 flip: sint<1>>, !firrtl.bundle<f1: uint<1>, f2 flip: sint<2>>
}

/// Destination bitwidth must be greater than or equal to source bitwidth.
firrtl.module @bitwidth(in %a : !firrtl.uint<1>, out %b : !firrtl.uint<2>) {
  // CHECK: firrtl.connect %b, %a
  firrtl.connect %b, %a : !firrtl.uint<2>, !firrtl.uint<1>
}

firrtl.module @wires0(in %in : !firrtl.uint<1>, out %out : !firrtl.uint<1>) {
  %w = firrtl.wire : !firrtl.uint<1>
  // CHECK: firrtl.connect %w, %in : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: firrtl.connect %out, %w : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %w, %in : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %out, %w : !firrtl.uint<1>, !firrtl.uint<1>
}

firrtl.module @wires1(in %in : !firrtl.uint<1>, out %out : !firrtl.uint<1>) {
  %wf = firrtl.wire : !firrtl.uint<1>
  // CHECK: firrtl.connect %wf, %in : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: firrtl.connect %out, %wf : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %wf, %in : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %out, %wf : !firrtl.uint<1>, !firrtl.uint<1>
}

firrtl.module @wires2() {
  %w0 = firrtl.wire : !firrtl.uint<1>
  %w1 = firrtl.wire : !firrtl.uint<1>
  // CHECK: firrtl.connect %w0, %w1
  firrtl.connect %w0, %w1 : !firrtl.uint<1>, !firrtl.uint<1>
}

firrtl.module @wires3(out %out : !firrtl.uint<1>) {
  %wf = firrtl.wire : !firrtl.uint<1>
  // check that we can read from an output port
  // CHECK: firrtl.connect %wf, %out
  firrtl.connect %wf, %out : !firrtl.uint<1>, !firrtl.uint<1>
}

firrtl.module @wires4(in %in : !firrtl.uint<1>, out %out : !firrtl.uint<1>) {
  %w = firrtl.wire : !firrtl.bundle<a: uint<1>>
  %0 = firrtl.subfield %w[a] : !firrtl.bundle<a: uint<1>>
  // CHECK: firrtl.connect %0, %in : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: firrtl.connect %out, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %0, %in : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %out, %0 : !firrtl.uint<1>, !firrtl.uint<1>
}

firrtl.module @registers0(in %clock : !firrtl.clock, in %in : !firrtl.uint<1>, out %out : !firrtl.uint<1>) {
  %0 = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<1>
  // CHECK: firrtl.connect %0, %in : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: firrtl.connect %out, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %0, %in : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %out, %0 : !firrtl.uint<1>, !firrtl.uint<1>
}

firrtl.module @registers1(in %clock : !firrtl.clock) {
  %0 = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<1>
  %1 = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<1>
  // CHECK: firrtl.connect %0, %1
  firrtl.connect %0, %1 : !firrtl.uint<1>, !firrtl.uint<1>
}

firrtl.module @ConstClock(in %in : !firrtl.const.clock, out %out : !firrtl.const.clock) {
  // CHECK: firrtl.connect %out, %in : !firrtl.const.clock, !firrtl.const.clock
  firrtl.connect %out, %in : !firrtl.const.clock, !firrtl.const.clock
}

firrtl.module @ConstReset(in %in : !firrtl.const.reset, out %out : !firrtl.const.reset) {
  // CHECK: firrtl.connect %out, %in : !firrtl.const.reset, !firrtl.const.reset
  firrtl.connect %out, %in : !firrtl.const.reset, !firrtl.const.reset
}

firrtl.module @ConstAsyncReset(in %in : !firrtl.const.asyncreset, out %out : !firrtl.const.asyncreset) {
  // CHECK: firrtl.connect %out, %in : !firrtl.const.asyncreset, !firrtl.const.asyncreset
  firrtl.connect %out, %in : !firrtl.const.asyncreset, !firrtl.const.asyncreset
}

firrtl.module @ConstUInt(in %in : !firrtl.const.uint<2>, out %out : !firrtl.const.uint<2>) {
  // CHECK: firrtl.connect %out, %in : !firrtl.const.uint<2>, !firrtl.const.uint<2>
  firrtl.connect %out, %in : !firrtl.const.uint<2>, !firrtl.const.uint<2>
}

firrtl.module @ConstSInt(in %in : !firrtl.const.sint<2>, out %out : !firrtl.const.sint<2>) {
  // CHECK: firrtl.connect %out, %in : !firrtl.const.sint<2>, !firrtl.const.sint<2>
  firrtl.connect %out, %in : !firrtl.const.sint<2>, !firrtl.const.sint<2>
}

firrtl.module @ConstVec(in %in : !firrtl.const.vector<uint<1>, 3>, out %out : !firrtl.const.vector<uint<1>, 3>) {
  // CHECK: firrtl.connect %out, %in : !firrtl.const.vector<uint<1>, 3>, !firrtl.const.vector<uint<1>, 3>
  firrtl.connect %out, %in : !firrtl.const.vector<uint<1>, 3>, !firrtl.const.vector<uint<1>, 3>
}

firrtl.module @ConstBundle(in %in : !firrtl.const.bundle<a: uint<1>, b: sint<2>>, out %out : !firrtl.const.bundle<a: uint<1>, b: sint<2>>) {
  // CHECK: firrtl.connect %out, %in : !firrtl.const.bundle<a: uint<1>, b: sint<2>>, !firrtl.const.bundle<a: uint<1>, b: sint<2>>
  firrtl.connect %out, %in : !firrtl.const.bundle<a: uint<1>, b: sint<2>>, !firrtl.const.bundle<a: uint<1>, b: sint<2>>
}

firrtl.module @MixedConstBundle(in %in : !firrtl.bundle<a: uint<1>, b: const.sint<2>>, out %out : !firrtl.bundle<a: uint<1>, b: const.sint<2>>) {
  // CHECK: firrtl.connect %out, %in : !firrtl.bundle<a: uint<1>, b: const.sint<2>>, !firrtl.bundle<a: uint<1>, b: const.sint<2>>
  firrtl.connect %out, %in : !firrtl.bundle<a: uint<1>, b: const.sint<2>>, !firrtl.bundle<a: uint<1>, b: const.sint<2>>
}

firrtl.module @ConstToExplicitConstElementsBundle(in %in : !firrtl.const.bundle<a: uint<1>, b: sint<2>>, out %out : !firrtl.const.bundle<a: const.uint<1>, b: const.sint<2>>) {
  // CHECK: firrtl.connect %out, %in : !firrtl.const.bundle<a: const.uint<1>, b: const.sint<2>>, !firrtl.const.bundle<a: uint<1>, b: sint<2>>
  firrtl.connect %out, %in : !firrtl.const.bundle<a: const.uint<1>, b: const.sint<2>>, !firrtl.const.bundle<a: uint<1>, b: sint<2>>
}

firrtl.module @ConstToNonConstClock(in %in : !firrtl.const.clock, out %out : !firrtl.clock) {
  // CHECK: firrtl.connect %out, %in : !firrtl.clock, !firrtl.const.clock
  firrtl.connect %out, %in : !firrtl.clock, !firrtl.const.clock
}

firrtl.module @ConstToNonConstReset(in %in : !firrtl.const.reset, out %out : !firrtl.reset) {
  // CHECK: firrtl.connect %out, %in : !firrtl.reset, !firrtl.const.reset
  firrtl.connect %out, %in : !firrtl.reset, !firrtl.const.reset
}

firrtl.module @ConstToNonConstAsyncReset(in %in : !firrtl.const.asyncreset, out %out : !firrtl.asyncreset) {
  // CHECK: firrtl.connect %out, %in : !firrtl.asyncreset, !firrtl.const.asyncreset
  firrtl.connect %out, %in : !firrtl.asyncreset, !firrtl.const.asyncreset
}

firrtl.module @ConstToNonConstUInt(in %in : !firrtl.const.uint<2>, out %out : !firrtl.uint<2>) {
  // CHECK: firrtl.connect %out, %in : !firrtl.uint<2>, !firrtl.const.uint<2>
  firrtl.connect %out, %in : !firrtl.uint<2>, !firrtl.const.uint<2>
}

firrtl.module @ConstToNonConstBundle(in %in : !firrtl.const.bundle<a: uint<1>, b: sint<2>>, out %out : !firrtl.bundle<a: uint<1>, b: sint<2>>) {
  // CHECK: firrtl.connect %out, %in : !firrtl.bundle<a: uint<1>, b: sint<2>>, !firrtl.const.bundle<a: uint<1>, b: sint<2>>
  firrtl.connect %out, %in : !firrtl.bundle<a: uint<1>, b: sint<2>>, !firrtl.const.bundle<a: uint<1>, b: sint<2>>
}

firrtl.module @MixedConstToNonConstBundle(in %in : !firrtl.bundle<a: uint<1>, b: const.sint<2>>, out %out : !firrtl.bundle<a: uint<1>, b: sint<2>>) {
  // CHECK: firrtl.connect %out, %in : !firrtl.bundle<a: uint<1>, b: sint<2>>, !firrtl.bundle<a: uint<1>, b: const.sint<2>>
  firrtl.connect %out, %in : !firrtl.bundle<a: uint<1>, b: sint<2>>, !firrtl.bundle<a: uint<1>, b: const.sint<2>>
}

firrtl.module @ConstToExplicitConstElementsVec(in %in : !firrtl.const.vector<uint<1>, 3>, out %out : !firrtl.vector<const.uint<1>, 3>) {
  // CHECK: firrtl.connect %out, %in : !firrtl.vector<const.uint<1>, 3>, !firrtl.const.vector<uint<1>, 3>
  firrtl.connect %out, %in : !firrtl.vector<const.uint<1>, 3>, !firrtl.const.vector<uint<1>, 3>
}

firrtl.module @ConstToNonConstVec(in %in : !firrtl.const.vector<uint<1>, 3>, out %out : !firrtl.vector<uint<1>, 3>) {
  // CHECK: firrtl.connect %out, %in : !firrtl.vector<uint<1>, 3>, !firrtl.const.vector<uint<1>, 3>
  firrtl.connect %out, %in : !firrtl.vector<uint<1>, 3>, !firrtl.const.vector<uint<1>, 3>
}

firrtl.module @NonConstToConstFlip(in %in   : !firrtl.bundle<a flip: uint<1>>,
                                   out %out : !firrtl.const.bundle<a flip: uint<1>>) {
  // CHECK: firrtl.connect %out, %in : !firrtl.const.bundle<a flip: uint<1>>, !firrtl.bundle<a flip: uint<1>>
  firrtl.connect %out, %in : !firrtl.const.bundle<a flip: uint<1>>, !firrtl.bundle<a flip: uint<1>>
}

firrtl.module @NonConstToNestedConstFlip(in %in   : !firrtl.bundle<a flip: uint<1>>,
                                         out %out : !firrtl.bundle<a flip: const.uint<1>>) {
  // CHECK: firrtl.connect %out, %in : !firrtl.bundle<a flip: const.uint<1>>, !firrtl.bundle<a flip: uint<1>>
  firrtl.connect %out, %in : !firrtl.bundle<a flip: const.uint<1>>, !firrtl.bundle<a flip: uint<1>>
}

firrtl.module @ConstToNonConstDoubleFlip(in %in   : !firrtl.const.bundle<a flip: bundle<a flip: uint<1>>>, 
                                         out %out : !firrtl.bundle<a flip: bundle<a flip: uint<1>>>) {
  // CHECK: firrtl.connect %out, %in :
  // CHECK-SAME: !firrtl.bundle<a flip: bundle<a flip: uint<1>>>, !firrtl.const.bundle<a flip: bundle<a flip: uint<1>>>
  firrtl.connect %out, %in : !firrtl.bundle<a flip: bundle<a flip: uint<1>>>, 
                             !firrtl.const.bundle<a flip: bundle<a flip: uint<1>>>
}

firrtl.module @NonConstToNonConstFlipFromConstSubaccess(in %in    : !firrtl.bundle<a flip: uint<1>>,
                                                        out %out  : !firrtl.const.vector<bundle<a flip: uint<1>>, 1>,
                                                        in %index : !firrtl.uint<1>) {
  %0 = firrtl.subaccess %out[%index] : !firrtl.const.vector<bundle<a flip: uint<1>>, 1>, !firrtl.uint<1>
  // CHECK: firrtl.connect %0, %in : !firrtl.bundle<a flip: uint<1>>, !firrtl.bundle<a flip: uint<1>>
  firrtl.connect %0, %in : !firrtl.bundle<a flip: uint<1>>, !firrtl.bundle<a flip: uint<1>>
}

// Const connections can occur within const-conditioned whens
// CHECK-LABEL: firrtl.module @ConstConditionConstAssign
firrtl.module @ConstConditionConstAssign(in %cond: !firrtl.const.uint<1>, in %in1: !firrtl.const.sint<2>, in %in2: !firrtl.const.sint<2>, out %out: !firrtl.const.sint<2>) {
  firrtl.when %cond : !firrtl.const.uint<1> {
    firrtl.strictconnect %out, %in1 : !firrtl.const.sint<2>
  } else {
    firrtl.strictconnect %out, %in2 : !firrtl.const.sint<2>
  }
}

// Non-const connections can occur within const-conditioned whens
// CHECK-LABEL: firrtl.module @ConstConditionNonConstAssign
firrtl.module @ConstConditionNonConstAssign(in %cond: !firrtl.const.uint<1>, in %in1: !firrtl.sint<2>, in %in2: !firrtl.sint<2>, out %out: !firrtl.sint<2>) {
  firrtl.when %cond : !firrtl.const.uint<1> {
    firrtl.strictconnect %out, %in1 : !firrtl.sint<2>
  } else {
    firrtl.strictconnect %out, %in2 : !firrtl.sint<2>
  }
}

// Const connections can occur when the destination is local to a non-const conditioned when block
// CHECK-LABEL: firrtl.module @NonConstWhenLocalConstAssign
firrtl.module @NonConstWhenLocalConstAssign(in %cond: !firrtl.uint<1>) {
  firrtl.when %cond : !firrtl.uint<1> {
    %w = firrtl.wire : !firrtl.const.uint<9>
    %c = firrtl.constant 0 : !firrtl.const.uint<9>
    firrtl.strictconnect %w, %c : !firrtl.const.uint<9>
  }
}

// Const connections can occur when the destination is local to a non-const 
// conditioned when block and the connection is inside a const conditioned when block
// CHECK-LABEL: firrtl.module @NonConstWhenLocalConstNestedConstWhenAssign
firrtl.module @NonConstWhenLocalConstNestedConstWhenAssign(in %cond: !firrtl.uint<1>, in %constCond: !firrtl.const.uint<1>) {
  firrtl.when %cond : !firrtl.uint<1> {
    %w = firrtl.wire : !firrtl.const.uint<9>
    firrtl.when %constCond : !firrtl.const.uint<1> {
      %c = firrtl.constant 0 : !firrtl.const.uint<9>
      firrtl.strictconnect %w, %c : !firrtl.const.uint<9>
    } else {
      %c = firrtl.constant 1 : !firrtl.const.uint<9>
      firrtl.strictconnect %w, %c : !firrtl.const.uint<9>
    }
  }
}

// Connections to flip const destinations are allowed within non-const when blocks 
// when the flow is to a non-const source
firrtl.module @NonConstWhenConstFlipAssign(in %p: !firrtl.uint<1>, in %in: !firrtl.bundle<a flip: uint<2>>, out %out: !firrtl.const.bundle<a flip: uint<2>>) {
  firrtl.when %p : !firrtl.uint<1> {
    firrtl.connect %out, %in : !firrtl.const.bundle<a flip: uint<2>>, !firrtl.bundle<a flip: uint<2>>
  }
}

// Connections to flip const destinations are allowed within non-const when blocks 
// when the flow is to a non-const source
firrtl.module @NonConstWhenNestedConstFlipAssign(in %p: !firrtl.uint<1>, in %in: !firrtl.bundle<a flip: uint<2>>, out %out: !firrtl.bundle<a flip: const.uint<2>>) {
  firrtl.when %p : !firrtl.uint<1> {
    firrtl.connect %out, %in : !firrtl.bundle<a flip: const.uint<2>>, !firrtl.bundle<a flip: uint<2>>
  }
}

// Connections to const flip sources can occur when the source is local to a non-const conditioned when block
// CHECK-LABEL: firrtl.module @NonConstWhenLocalConstFlipAssign
firrtl.module @NonConstWhenLocalConstFlipAssign(in %cond: !firrtl.uint<1>, out %out : !firrtl.const.bundle<a flip: uint<2>>) {
  firrtl.when %cond : !firrtl.uint<1> {
    %w = firrtl.wire : !firrtl.const.bundle<a flip: uint<2>>
    firrtl.connect %out, %w : !firrtl.const.bundle<a flip: uint<2>>, !firrtl.const.bundle<a flip: uint<2>>
  }
}

// Connections to nested const flip sources can occur when the source is local to a non-const conditioned when block
// CHECK-LABEL: firrtl.module @NonConstWhenLocalNestedConstFlipAssign
firrtl.module @NonConstWhenLocalNestedConstFlipAssign(in %cond: !firrtl.uint<1>, out %out : !firrtl.bundle<a flip: const.uint<2>>) {
  firrtl.when %cond : !firrtl.uint<1> {
    %w = firrtl.wire : !firrtl.bundle<a flip: const.uint<2>>
    firrtl.connect %out, %w : !firrtl.bundle<a flip: const.uint<2>>, !firrtl.bundle<a flip: const.uint<2>>
  }
}
}
