// RUN: firtool --hw --split-input-file --verify-diagnostics %s
// These will be picked up by https://github.com/llvm/circt/pull/1444

// Tests extracted from:
// - test/scala/firrtlTests/AsyncResetSpec.scala

firrtl.circuit "Foo" {
  // expected-note @+1 {{reset value defined here:}}
  firrtl.module @Foo(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %v: !firrtl.uint<8>) {
    // Constant check should see through subfield connects.
    %bundle0 = firrtl.wire : !firrtl.bundle<a: uint<8>>
    %bundle0.a = firrtl.subfield %bundle0[0] : !firrtl.bundle<a: uint<8>>
    firrtl.connect %bundle0.a, %v : !firrtl.uint<8>, !firrtl.uint<8>
    // expected-error @+2 {{register with async reset requires constant reset value}}
    // expected-error @+1 {{'firrtl.regreset' op LowerToHW couldn't handle this operation}}
    %2 = firrtl.regreset %clock, %reset, %bundle0 : !firrtl.clock, !firrtl.asyncreset, !firrtl.bundle<a: uint<8>>, !firrtl.bundle<a: uint<8>>
  }
}

// -----

firrtl.circuit "Foo" {
  // expected-note @+1 {{reset value defined here:}}
  firrtl.module @Foo(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %v: !firrtl.uint<8>) {
    // Constant check should see through multiple connect hops.
    %bundle0 = firrtl.wire : !firrtl.bundle<a: uint<8>>
    %bundle0.a = firrtl.subfield %bundle0[0] : !firrtl.bundle<a: uint<8>>
    firrtl.connect %bundle0.a, %v : !firrtl.uint<8>, !firrtl.uint<8>
    %bundle1 = firrtl.wire : !firrtl.bundle<a: uint<8>>
    firrtl.connect %bundle1, %bundle0 : !firrtl.bundle<a: uint<8>>, !firrtl.bundle<a: uint<8>>
    // expected-error @+2 {{register with async reset requires constant reset value}}
    // expected-error @+1 {{'firrtl.regreset' op LowerToHW couldn't handle this operation}}
    %3 = firrtl.regreset %clock, %reset, %bundle1 : !firrtl.clock, !firrtl.asyncreset, !firrtl.bundle<a: uint<8>>, !firrtl.bundle<a: uint<8>>
  }
}

// -----

firrtl.circuit "Foo" {
  // expected-note @+1 {{reset value defined here:}}
  firrtl.module @Foo(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %v: !firrtl.uint<8>) {
    // Constant check should see through subindex connects.
    %vector0 = firrtl.wire : !firrtl.vector<uint<8>, 1>
    %vector0.a = firrtl.subindex %vector0[0] : !firrtl.vector<uint<8>, 1>
    firrtl.connect %vector0.a, %v : !firrtl.uint<8>, !firrtl.uint<8>
    // expected-error @+2 {{register with async reset requires constant reset value}}
    // expected-error @+1 {{'firrtl.regreset' op LowerToHW couldn't handle this operation}}
    %4 = firrtl.regreset %clock, %reset, %vector0 : !firrtl.clock, !firrtl.asyncreset, !firrtl.vector<uint<8>, 1>, !firrtl.vector<uint<8>, 1>
  }
}

// -----

firrtl.circuit "Foo" {
  // expected-note @+1 {{reset value defined here:}}
  firrtl.module @Foo(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %v: !firrtl.uint<8>) {
    // Constant check should see through multiple connect hops.
    %vector0 = firrtl.wire : !firrtl.vector<uint<8>, 1>
    %vector0.a = firrtl.subindex %vector0[0] : !firrtl.vector<uint<8>, 1>
    firrtl.connect %vector0.a, %v : !firrtl.uint<8>, !firrtl.uint<8>
    %vector1 = firrtl.wire : !firrtl.vector<uint<8>, 1>
    firrtl.connect %vector1, %vector0 : !firrtl.vector<uint<8>, 1>, !firrtl.vector<uint<8>, 1>
    // expected-error @+2 {{register with async reset requires constant reset value}}
    // expected-error @+1 {{'firrtl.regreset' op LowerToHW couldn't handle this operation}}
    %5 = firrtl.regreset %clock, %reset, %vector1 : !firrtl.clock, !firrtl.asyncreset, !firrtl.vector<uint<8>, 1>, !firrtl.vector<uint<8>, 1>
  }
}

// -----

// Hidden Non-literals should NOT be allowed as reset values for AsyncReset
firrtl.circuit "Foo" {
  // expected-note @+1 {{reset value defined here:}}
  firrtl.module @Foo(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %x: !firrtl.vector<uint<1>, 4>, in %y: !firrtl.uint<1>, out %z: !firrtl.vector<uint<1>, 4>) {
    %literal = firrtl.wire  : !firrtl.vector<uint<1>, 4>
    %0 = firrtl.subindex %literal[0] : !firrtl.vector<uint<1>, 4>
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    firrtl.connect %0, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %1 = firrtl.subindex %literal[1] : !firrtl.vector<uint<1>, 4>
    firrtl.connect %1, %y : !firrtl.uint<1>, !firrtl.uint<1>
    %2 = firrtl.subindex %literal[2] : !firrtl.vector<uint<1>, 4>
    firrtl.connect %2, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %3 = firrtl.subindex %literal[3] : !firrtl.vector<uint<1>, 4>
    firrtl.connect %3, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    // expected-error @+2 {{register with async reset requires constant reset value}}
    // expected-error @+1 {{'firrtl.regreset' op LowerToHW couldn't handle this operation}}
    %r = firrtl.regreset %clock, %reset, %literal  : !firrtl.clock, !firrtl.asyncreset, !firrtl.vector<uint<1>, 4>, !firrtl.vector<uint<1>, 4>
    firrtl.connect %r, %x : !firrtl.vector<uint<1>, 4>, !firrtl.vector<uint<1>, 4>
    firrtl.connect %z, %r : !firrtl.vector<uint<1>, 4>, !firrtl.vector<uint<1>, 4>
  }
}

// -----

// Wire connected to non-literal should NOT be allowed as reset values for AsyncReset
firrtl.circuit "Foo"   {
  firrtl.module @Foo(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %x: !firrtl.uint<1>, in %y: !firrtl.uint<1>, in %cond: !firrtl.uint<1>, out %z: !firrtl.uint<1>) {
    // expected-note @+1 {{reset value defined here:}}
    %w = firrtl.wire  : !firrtl.uint<1>
    %c1_ui = firrtl.constant 1 : !firrtl.uint
    firrtl.connect %w, %c1_ui : !firrtl.uint<1>, !firrtl.uint
    firrtl.when %cond : !firrtl.uint<1> {
      firrtl.connect %w, %y : !firrtl.uint<1>, !firrtl.uint<1>
    }
    // expected-error @+2 {{register with async reset requires constant reset value}}
    // expected-error @+1 {{'firrtl.regreset' op LowerToHW couldn't handle this operation}}
    %r = firrtl.regreset %clock, %reset, %w  : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %r, %x : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %z, %r : !firrtl.uint<1>, !firrtl.uint<1>
  }
}
