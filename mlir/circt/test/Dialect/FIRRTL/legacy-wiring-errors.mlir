// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-annotations))' -split-input-file %s -verify-diagnostics

// Every Wiring pin must have exactly one defined source
//
// expected-error @+1 {{Unable to resolve source for pin: "foo_out"}}
firrtl.circuit "Foo" attributes {
  rawAnnotations = [
    {
      class = "firrtl.passes.wiring.SinkAnnotation",
      target = "Foo.Foo.out",
      pin = "foo_out"
    }]} {
  firrtl.module @Foo(out %out: !firrtl.uint<1>) {
      firrtl.skip
  }
}

// -----

// Every Wiring pin must have at least one defined sink
//
// expected-error @+1 {{Unable to resolve sink(s) for pin: "foo_in"}}
firrtl.circuit "Foo" attributes {
  rawAnnotations = [
    {
      class = "firrtl.passes.wiring.SourceAnnotation",
      target = "Foo.Foo.in",
      pin = "foo_in"
    }]} {
  firrtl.module @Foo(in %in: !firrtl.uint<1>) {
      firrtl.skip
  }
}

// -----

// Multiple SourceAnnotations for the same pin are forbidden
//
// expected-error @+2 {{Unable to apply annotation: {class = "firrtl.passes.wiring.SourceAnnotation", pin = "foo_out", target = "Foo.Foo.b"}}}
// expected-error @+1 {{More than one firrtl.passes.wiring.SourceAnnotation defined for pin "foo_out"}}
firrtl.circuit "Foo" attributes {
  rawAnnotations = [
    {
      class = "firrtl.passes.wiring.SinkAnnotation",
      target = "Foo.Foo.out",
      pin = "foo_out"
    },
    {
      class = "firrtl.passes.wiring.SourceAnnotation",
      target = "Foo.Foo.a",
      pin = "foo_out"
    },
    {
      class = "firrtl.passes.wiring.SourceAnnotation",
      target = "Foo.Foo.b",
      pin = "foo_out"
    }]} {
  firrtl.module @Foo(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
      firrtl.skip
  }
}

// -----
// Error if attempt to wire incompatible types.

firrtl.circuit "Foo" attributes {
 rawAnnotations = [
  {
    class = "firrtl.passes.wiring.SourceAnnotation",
    target = "~Foo|Bar>y",
    pin = "xyz"
  },
  {
    class = "firrtl.passes.wiring.SinkAnnotation",
    target = "~Foo|Foo>x",
    pin = "xyz"
  }
  ]} {
  firrtl.module private @Bar() {
    // expected-error @below {{Wiring Problem source type '!firrtl.bundle<a: uint<1>, b: uint<2>>' does not match sink type '!firrtl.uint<1>'}}
    %y = firrtl.wire interesting_name : !firrtl.bundle<a: uint<1>, b: uint<2>>
    %invalid_reset = firrtl.invalidvalue : !firrtl.bundle<a: uint<1>, b: uint<2>>
    firrtl.strictconnect %y, %invalid_reset : !firrtl.bundle<a: uint<1>, b: uint<2>>
  }
  firrtl.module @Foo() {
    firrtl.instance bar interesting_name @Bar()
    // expected-note @below {{The sink is here.}}
    %x = firrtl.wire interesting_name : !firrtl.uint<1>
    %invalid_ui1 = firrtl.invalidvalue : !firrtl.uint<1>
    firrtl.strictconnect %x, %invalid_ui1 : !firrtl.uint<1>
  }
}
