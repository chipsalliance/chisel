// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-infer-resets))' --verify-diagnostics --split-input-file %s

// Tests extracted from:
// - github.com/chipsalliance/firrtl:
//   - test/scala/firrtlTests/InferResetsSpec.scala
// - github.com/sifive/$internal:
//   - test/scala/firrtl/FullAsyncResetTransform.scala


//===----------------------------------------------------------------------===//
// Reset Inference
//===----------------------------------------------------------------------===//


// Should NOT allow last connect semantics to pick the right type for Reset
firrtl.circuit "top" {
  // expected-error @+2 {{reset network "reset0" simultaneously connected to async and sync resets}}
  // expected-note @+1 {{majority of connections to this reset are async}}
  firrtl.module @top(in %reset0: !firrtl.asyncreset, in %reset1: !firrtl.uint<1>, out %out: !firrtl.reset) {
    %w0 = firrtl.wire : !firrtl.reset
    %w1 = firrtl.wire : !firrtl.reset
    firrtl.connect %w0, %reset0 : !firrtl.reset, !firrtl.asyncreset
    // expected-note @+1 {{sync drive here:}}
    firrtl.connect %w1, %reset1 : !firrtl.reset, !firrtl.uint<1>
    firrtl.connect %out, %w0 : !firrtl.reset, !firrtl.reset
    firrtl.connect %out, %w1 : !firrtl.reset, !firrtl.reset
  }
}

// -----
// Should NOT support last connect semantics across whens
firrtl.circuit "top" {
  // expected-error @+2 {{reset network "reset2" simultaneously connected to async and sync resets}}
  // expected-note @+1 {{majority of connections to this reset are async}}
  firrtl.module @top(in %reset0: !firrtl.asyncreset, in %reset1: !firrtl.asyncreset, in %reset2: !firrtl.uint<1>, in %en: !firrtl.uint<1>, out %out: !firrtl.reset) {
    %w0 = firrtl.wire : !firrtl.reset
    %w1 = firrtl.wire : !firrtl.reset
    %w2 = firrtl.wire : !firrtl.reset
    firrtl.connect %w0, %reset0 : !firrtl.reset, !firrtl.asyncreset
    firrtl.connect %w1, %reset1 : !firrtl.reset, !firrtl.asyncreset
    // expected-note @+1 {{sync drive here:}}
    firrtl.connect %w2, %reset2 : !firrtl.reset, !firrtl.uint<1>
    firrtl.connect %out, %w2 : !firrtl.reset, !firrtl.reset
    firrtl.when %en : !firrtl.uint<1>  {
      firrtl.connect %out, %w0 : !firrtl.reset, !firrtl.reset
    } else  {
      firrtl.connect %out, %w1 : !firrtl.reset, !firrtl.reset
    }
  }
}

// -----
// Should not allow different Reset Types to drive a single Reset
firrtl.circuit "top" {
  // expected-error @+2 {{reset network "reset0" simultaneously connected to async and sync resets}}
  // expected-note @+1 {{majority of connections to this reset are async}}
  firrtl.module @top(in %reset0: !firrtl.asyncreset, in %reset1: !firrtl.uint<1>, in %en: !firrtl.uint<1>, out %out: !firrtl.reset) {
    %w1 = firrtl.wire : !firrtl.reset
    %w2 = firrtl.wire : !firrtl.reset
    firrtl.connect %w1, %reset0 : !firrtl.reset, !firrtl.asyncreset
    // expected-note @+1 {{sync drive here:}}
    firrtl.connect %w2, %reset1 : !firrtl.reset, !firrtl.uint<1>
    firrtl.connect %out, %w1 : !firrtl.reset, !firrtl.reset
    firrtl.when %en : !firrtl.uint<1>  {
      firrtl.connect %out, %w2 : !firrtl.reset, !firrtl.reset
    }
  }
}

// -----
// Should error if a ResetType driving UInt<1> infers to AsyncReset
firrtl.circuit "top" {
  // expected-error @+2 {{reset network "in" simultaneously connected to async and sync resets}}
  // expected-note @+1 {{majority of connections to this reset are async}}
  firrtl.module @top(in %in: !firrtl.asyncreset, out %out: !firrtl.uint<1>) {
    %w = firrtl.wire  : !firrtl.reset
    firrtl.connect %w, %in : !firrtl.reset, !firrtl.asyncreset
    // expected-note @+1 {{sync drive here:}}
    firrtl.connect %out, %w : !firrtl.uint<1>, !firrtl.reset
  }
}

// -----
// Should error if a ResetType driving AsyncReset infers to UInt<1>
firrtl.circuit "top"   {
  // expected-error @+2 {{reset network "in" simultaneously connected to async and sync resets}}
  // expected-note @+1 {{majority of connections to this reset are async}}
  firrtl.module @top(in %in: !firrtl.uint<1>, out %out: !firrtl.asyncreset) {
    %w = firrtl.wire  : !firrtl.reset
    // expected-note @+1 {{sync drive here:}}
    firrtl.connect %w, %in : !firrtl.reset, !firrtl.uint<1>
    firrtl.connect %out, %w : !firrtl.asyncreset, !firrtl.reset
  }
}

// -----
// Should not allow ResetType as an Input
firrtl.circuit "top" {
  // expected-error @+2 {{reset network never driven with concrete type}}
  // expected-note @+1 {{here: }}
  firrtl.module @top(in %in: !firrtl.bundle<foo: reset>, out %out: !firrtl.reset) {
    // expected-note @+1 {{here: }}
    %0 = firrtl.subfield %in[foo] : !firrtl.bundle<foo: reset>
    firrtl.connect %out, %0 : !firrtl.reset, !firrtl.reset
  }
}

// -----
// Should not allow ResetType as an ExtModule output
firrtl.circuit "top" {
  firrtl.extmodule @ext(out out: !firrtl.bundle<foo: reset>)
  // expected-note @+1 {{here: }}
  firrtl.module @top(out %out: !firrtl.reset) {
    // expected-error @+2 {{reset network never driven with concrete type}}
    // expected-note @+1 {{here: }}
    %e_out = firrtl.instance e @ext(out out: !firrtl.bundle<foo: reset>)
    // expected-note @+1 {{here: }}
    %0 = firrtl.subfield %e_out[foo] : !firrtl.bundle<foo: reset>
    firrtl.connect %out, %0 : !firrtl.reset, !firrtl.reset
  }
}

// -----
// Should not allow Vecs to infer different Reset Types
firrtl.circuit "top" {
  // expected-error @+2 {{reset network "out[]" simultaneously connected to async and sync resets}}
  // expected-note @+1 {{majority of connections to this reset are async}}
  firrtl.module @top(in %reset0: !firrtl.asyncreset, in %reset1: !firrtl.uint<1>, out %out: !firrtl.vector<reset, 2>) {
    %0 = firrtl.subindex %out[0] : !firrtl.vector<reset, 2>
    %1 = firrtl.subindex %out[1] : !firrtl.vector<reset, 2>
    firrtl.connect %0, %reset0 : !firrtl.reset, !firrtl.asyncreset
    // expected-note @+1 {{sync drive here:}}
    firrtl.connect %1, %reset1 : !firrtl.reset, !firrtl.uint<1>
  }
}

// -----
// Should not allow an invalidated Wire to drive both a UInt<1> and an AsyncReset
firrtl.circuit "top" {
  // expected-error @+2 {{reset network "in0" simultaneously connected to async and sync resets}}
  // expected-note @+1 {{majority of connections to this reset are async}}
  firrtl.module @top(in %in0: !firrtl.asyncreset, in %in1: !firrtl.uint<1>, out %out0: !firrtl.reset, out %out1: !firrtl.reset) {
    %w = firrtl.wire  : !firrtl.reset
    %invalid_reset = firrtl.invalidvalue : !firrtl.reset
    firrtl.connect %w, %invalid_reset : !firrtl.reset, !firrtl.reset
    firrtl.connect %out0, %w : !firrtl.reset, !firrtl.reset
    firrtl.connect %out1, %w : !firrtl.reset, !firrtl.reset
    firrtl.connect %out0, %in0 : !firrtl.reset, !firrtl.asyncreset
    // expected-note @+1 {{sync drive here:}}
    firrtl.connect %out1, %in1 : !firrtl.reset, !firrtl.uint<1>
  }
}

//===----------------------------------------------------------------------===//
// Full Async Reset
//===----------------------------------------------------------------------===//

// -----
// Reset annotation cannot target module
firrtl.circuit "top" {
  // expected-error @+1 {{FullAsyncResetAnnotation' cannot target module; must target port or wire/node instead}}
  firrtl.module @top() attributes {annotations = [{class = "sifive.enterprise.firrtl.FullAsyncResetAnnotation"}]} {
  }
}

// -----
// Ignore reset annotation cannot target port
firrtl.circuit "top" {
  // expected-error @+1 {{IgnoreFullAsyncResetAnnotation' cannot target port; must target module instead}}
  firrtl.module @top(in %reset: !firrtl.asyncreset) attributes {portAnnotations =[[{class = "sifive.enterprise.firrtl.IgnoreFullAsyncResetAnnotation"}]]} {
  }
}

// -----
// Ignore reset annotation cannot target wire/node
firrtl.circuit "top" {
  firrtl.module @top() {
    // expected-error @+1 {{IgnoreFullAsyncResetAnnotation' cannot target wire/node; must target module instead}}
    %0 = firrtl.wire {annotations = [{class = "sifive.enterprise.firrtl.IgnoreFullAsyncResetAnnotation"}]} : !firrtl.asyncreset
    // expected-error @+1 {{IgnoreFullAsyncResetAnnotation' cannot target wire/node; must target module instead}}
    %1 = firrtl.node %0 {annotations = [{class = "sifive.enterprise.firrtl.IgnoreFullAsyncResetAnnotation"}]} : !firrtl.asyncreset
    // expected-error @+1 {{reset annotations must target module, port, or wire/node}}
    %2 = firrtl.asUInt %0 {annotations = [{class = "sifive.enterprise.firrtl.FullAsyncResetAnnotation"}]} : (!firrtl.asyncreset) -> !firrtl.uint<1>
    // expected-error @+1 {{reset annotations must target module, port, or wire/node}}
    %3 = firrtl.asUInt %0 {annotations = [{class = "sifive.enterprise.firrtl.IgnoreFullAsyncResetAnnotation"}]} : (!firrtl.asyncreset) -> !firrtl.uint<1>
  }
}

// -----
// Cannot have multiple reset annotations on a module
firrtl.circuit "top" {
  // expected-error @+2 {{multiple reset annotations on module 'top'}}
  // expected-note @+1 {{conflicting "sifive.enterprise.firrtl.FullAsyncResetAnnotation":}}
  firrtl.module @top(in %outerReset: !firrtl.asyncreset) attributes {portAnnotations = [[{class = "sifive.enterprise.firrtl.FullAsyncResetAnnotation"}]]} {
    // expected-note @+1 {{conflicting "sifive.enterprise.firrtl.FullAsyncResetAnnotation":}}
    %innerReset = firrtl.wire {annotations = [{class = "sifive.enterprise.firrtl.FullAsyncResetAnnotation"}]} : !firrtl.asyncreset
    // expected-note @+1 {{conflicting "sifive.enterprise.firrtl.FullAsyncResetAnnotation":}}
    %anotherReset = firrtl.node %innerReset {annotations = [{class = "sifive.enterprise.firrtl.FullAsyncResetAnnotation"}]} : !firrtl.asyncreset
  }
}

// -----
// Multiple instances of same module cannot live in different reset domains
firrtl.circuit "Top" {
  // expected-error @+1 {{module 'Foo' instantiated in different reset domains}}
  firrtl.module @Foo(in %clock: !firrtl.clock) {
    %reg = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<8>
  }
  // expected-note @+1 {{reset domain 'otherReset' of module 'Child' declared here:}}
  firrtl.module @Child(in %clock: !firrtl.clock, in %otherReset: !firrtl.asyncreset) attributes {portAnnotations = [[],[{class = "sifive.enterprise.firrtl.FullAsyncResetAnnotation"}]]} {
    // expected-note @+1 {{instance 'child/inst' is in reset domain rooted at 'otherReset' of module 'Child'}}
    %inst_clock = firrtl.instance inst @Foo(in clock: !firrtl.clock)
    firrtl.connect %inst_clock, %clock : !firrtl.clock, !firrtl.clock
  }
  firrtl.module @Other(in %clock: !firrtl.clock) attributes {annotations = [{class = "sifive.enterprise.firrtl.IgnoreFullAsyncResetAnnotation"}]} {
    // expected-note @+1 {{instance 'other/inst' is in no reset domain}}
    %inst_clock = firrtl.instance inst @Foo(in clock: !firrtl.clock)
    firrtl.connect %inst_clock, %clock : !firrtl.clock, !firrtl.clock
  }
  // expected-note @+1 {{reset domain 'reset' of module 'Top' declared here:}}
  firrtl.module @Top(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset) attributes {portAnnotations = [[],[{class = "sifive.enterprise.firrtl.FullAsyncResetAnnotation"}]]} {
    %child_clock, %child_otherReset = firrtl.instance child @Child(in clock: !firrtl.clock, in otherReset: !firrtl.asyncreset)
    %other_clock = firrtl.instance other @Other(in clock: !firrtl.clock)
    // expected-note @+1 {{instance 'foo' is in reset domain rooted at 'reset' of module 'Top'}}
    %foo_clock = firrtl.instance foo @Foo(in clock: !firrtl.clock)
    firrtl.connect %child_clock, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %other_clock, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %foo_clock, %clock : !firrtl.clock, !firrtl.clock
  }
}

// -----

firrtl.circuit "UninferredReset" {
  // expected-error @+2 {{a port "reset" with abstract reset type was unable to be inferred by InferResets}}
  // expected-note @+1 {{the module with this uninferred reset port was defined here}}
  firrtl.module @UninferredReset(in %reset: !firrtl.reset) {}
}

// -----

firrtl.circuit "UninferredRefReset" {
  // expected-error @+2 {{a port "reset" with abstract reset type was unable to be inferred by InferResets}}
  // expected-note @+1 {{the module with this uninferred reset port was defined here}}
  firrtl.module @UninferredRefReset(in %reset: !firrtl.probe<reset>) {}
}
