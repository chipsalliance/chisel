// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-annotations))' --split-input-file %s | FileCheck %s

// Check added ports are real type
// CHECK-LABEL: firrtl.circuit "FooBar"
firrtl.circuit "FooBar" attributes {
  rawAnnotations = [
    {
      class = "firrtl.passes.wiring.SinkAnnotation",
      target = "FooBar.Foo.io.out",
      pin = "foo_out"
    },
    {
      class = "firrtl.passes.wiring.SourceAnnotation",
      target = "FooBar.FooBar.io.in",
      pin = "foo_out"
    }]} {
  // CHECK: firrtl.module @Foo
  // The real port type of the source should be bored
  // CHECK-SAME: in %io_out__bore: !firrtl.uint<1>
  firrtl.module@Foo(out %io: !firrtl.bundle<out: uint<1>>) {
      firrtl.skip
  }
  // CHECK: firrtl.module @Bar
  // The real port type of the source should be bored in the parent
  // CHECK-SAME: in %foo_io_out__bore: !firrtl.uint<1>
  firrtl.module @Bar(out %io: !firrtl.bundle<out: uint<1>>) {
      %0 = firrtl.subfield %io[out] : !firrtl.bundle<out: uint<1>>
      // CHECK: firrtl.instance foo
      // CHECK-SAME: in io_out__bore: !firrtl.uint<1>
      %foo_io = firrtl.instance foo interesting_name  @Foo(out io: !firrtl.bundle<out: uint<1>>)
      %1 = firrtl.subfield %foo_io[out] : !firrtl.bundle<out: uint<1>>
      firrtl.strictconnect %0, %1 : !firrtl.uint<1>
  }
  // CHECK: firrtl.module @FooBar
  firrtl.module @FooBar(out %io: !firrtl.bundle<in flip: uint<1>, out: uint<1>>) {
      %0 = firrtl.subfield %io[out] : !firrtl.bundle<in flip: uint<1>, out: uint<1>>
      // CHECK: firrtl.instance bar
      // CHECK-SAME: in foo_io_out__bore: !firrtl.uint<1>
      %bar_io = firrtl.instance bar interesting_name  @Bar(out io: !firrtl.bundle<out: uint<1>>)
      %1 = firrtl.subfield %bar_io[out] : !firrtl.bundle<out: uint<1>>
      firrtl.strictconnect %0, %1 : !firrtl.uint<1>
  }
}

// -----

// Test the behaviour of single source, multiple sink
// CHECK-LABEL: firrtl.circuit "FooBar"
firrtl.circuit "FooBar" attributes {
  rawAnnotations = [
    {
      class = "firrtl.passes.wiring.SourceAnnotation",
      target = "FooBar.FooBar.io.in",
      pin = "in"
    },
    {
      class = "firrtl.passes.wiring.SinkAnnotation",
      target = "FooBar.Foo.io.out",
      pin = "in"
    },
    {
      class = "firrtl.passes.wiring.SinkAnnotation",
      target = "FooBar.Foo_1.io.out",
      pin = "in"
    },
    {
      class = "firrtl.passes.wiring.SinkAnnotation",
      target = "FooBar.Bar.io.out",
      pin = "in"
    }]} {
  // CHECK: firrtl.module @Foo
  // CHECK-SAME: in %io_out__bore: !firrtl.uint<1>
  firrtl.module @Foo(out %io: !firrtl.bundle<out: uint<1>>) {
    firrtl.skip
    // CHECK: %0 = firrtl.subfield %io[out] : !firrtl.bundle<out: uint<1>>
    // CHECK: firrtl.strictconnect %0, %io_out__bore : !firrtl.uint<1>
  }
  // CHECK: firrtl.module @Foo_1
  // CHECK-SAME: in %io_out__bore: !firrtl.uint<1>
  firrtl.module @Foo_1(out %io: !firrtl.bundle<out: uint<1>>) {
    firrtl.skip
    // CHECK: %0 = firrtl.subfield %io[out] : !firrtl.bundle<out: uint<1>>
    // CHECK: firrtl.strictconnect %0, %io_out__bore : !firrtl.uint<1>
  }
  // CHECK: firrtl.module @Bar
  // CHECK-SAME: in %io_out__bore: !firrtl.uint<1>
  firrtl.module @Bar(out %io: !firrtl.bundle<out: uint<1>>) {
    firrtl.skip
    // CHECK: %0 = firrtl.subfield %io[out] : !firrtl.bundle<out: uint<1>>
    // CHECK: firrtl.strictconnect %0, %io_out__bore : !firrtl.uint<1>
  }
  // CHECK: firrtl.module @FooBar
  firrtl.module @FooBar(out %io: !firrtl.bundle<in flip: uint<1>, out_foo0: uint<1>, out_foo1: uint<1>, out_bar: uint<1>>) {
    // CHECK: %0 = firrtl.subfield %io[in] : !firrtl.bundle<in flip: uint<1>, out_foo0: uint<1>, out_foo1: uint<1>, out_bar: uint<1>>
    %0 = firrtl.subfield %io[out_bar] : !firrtl.bundle<in flip: uint<1>, out_foo0: uint<1>, out_foo1: uint<1>, out_bar: uint<1>>
    %1 = firrtl.subfield %io[out_foo1] : !firrtl.bundle<in flip: uint<1>, out_foo0: uint<1>, out_foo1: uint<1>, out_bar: uint<1>>
    %2 = firrtl.subfield %io[out_foo0] : !firrtl.bundle<in flip: uint<1>, out_foo0: uint<1>, out_foo1: uint<1>, out_bar: uint<1>>
    // CHECK: firrtl.instance foo0
    // CHECK-SAME: in io_out__bore: !firrtl.uint<1>
    %foo0_io = firrtl.instance foo0 interesting_name  @Foo(out io: !firrtl.bundle<out: uint<1>>)
    %3 = firrtl.subfield %foo0_io[out] : !firrtl.bundle<out: uint<1>>
    // CHECK: firrtl.instance foo1
    // CHECK-SAME: in io_out__bore: !firrtl.uint<1>
    %foo1_io = firrtl.instance foo1 interesting_name  @Foo_1(out io: !firrtl.bundle<out: uint<1>>)
    %4 = firrtl.subfield %foo1_io[out] : !firrtl.bundle<out: uint<1>>
    // CHECK: firrtl.instance bar
    // CHECK-SAME: in io_out__bore: !firrtl.uint<1>
    %bar_io = firrtl.instance bar interesting_name  @Bar(out io: !firrtl.bundle<out: uint<1>>)
    %5 = firrtl.subfield %bar_io[out] : !firrtl.bundle<out: uint<1>>
    firrtl.strictconnect %2, %3 : !firrtl.uint<1>
    firrtl.strictconnect %1, %4 : !firrtl.uint<1>
    firrtl.strictconnect %0, %5 : !firrtl.uint<1>
    // CHECK: firrtl.strictconnect %foo0_io_out__bore, %0 : !firrtl.uint<1>
    // CHECK: firrtl.strictconnect %foo1_io_out__bore, %0 : !firrtl.uint<1>
    // CHECK: firrtl.strictconnect %bar_io_out__bore, %0 : !firrtl.uint<1>
  }
}

// -----

// CHECK-LABEL: circuit "Sub"
firrtl.circuit "Sub" attributes {
  rawAnnotations = [
    {
      class = "firrtl.passes.wiring.SourceAnnotation",
      target = "Sub.Sub.a[0]",
      pin = "test"
    },
    {
      class = "firrtl.passes.wiring.SinkAnnotation",
      target = "Sub.Sub.b[0]",
      pin = "test"
    }]} {
  firrtl.module @Sub() {
    // CHECK:      %[[a:.+]] = firrtl.wire
    // CHECK-NEXT: %[[a_0:.+]] = firrtl.subindex %[[a]][0]
    // CHECK:      %[[b:.+]] = firrtl.wire
    // CHECK-NEXT: %[[b_0:.+]] = firrtl.subindex %[[b]][0]
    %a = firrtl.wire interesting_name : !firrtl.vector<uint<1>,1>
    %b = firrtl.wire interesting_name : !firrtl.vector<uint<1>,1>
  }
}

// -----

// https://github.com/llvm/circt/issues/4651
// Check that wiring can convert compatible types that can normally be connected.

firrtl.circuit "ResetToI1" attributes {
 rawAnnotations = [
  {
    class = "firrtl.passes.wiring.SourceAnnotation",
    target = "~ResetToI1|Bar>y",
    pin = "xyz"
  },
  {
    class = "firrtl.passes.wiring.SinkAnnotation",
    target = "~ResetToI1|ResetToI1>x",
    pin = "xyz"
  }
  ]} {
  firrtl.module private @Bar() {
    %y = firrtl.wire interesting_name : !firrtl.reset
    %invalid_reset = firrtl.invalidvalue : !firrtl.reset
    firrtl.strictconnect %y, %invalid_reset : !firrtl.reset
  }
  // CHECK-LABEL: module @ResetToI1
  firrtl.module @ResetToI1() {
    // CHECK: %[[r1:.+]] = firrtl.resetCast %{{[^ ]*}}
    // CHECK-NEXT: firrtl.strictconnect %x, %[[r1]] : !firrtl.uint<1>
    firrtl.instance bar interesting_name @Bar()
    %x = firrtl.wire interesting_name : !firrtl.uint<1>
    %invalid_ui1 = firrtl.invalidvalue : !firrtl.uint<1>
    firrtl.strictconnect %x, %invalid_ui1 : !firrtl.uint<1>
  }
}

// -----

// Similarly, but for integer types.

firrtl.circuit "IntWidths" attributes {
 rawAnnotations = [
  {
    class = "firrtl.passes.wiring.SourceAnnotation",
    target = "~IntWidths|Bar>y",
    pin = "xyz"
  },
  {
    class = "firrtl.passes.wiring.SinkAnnotation",
    target = "~IntWidths|IntWidths>x",
    pin = "xyz"
  }
  ]} {
  firrtl.module private @Bar() {
    %y = firrtl.wire interesting_name : !firrtl.uint<4>
    %invalid_reset = firrtl.invalidvalue : !firrtl.uint<4>
    firrtl.strictconnect %y, %invalid_reset : !firrtl.uint<4>
  }
  // CHECK-LABEL: module @IntWidths
  firrtl.module @IntWidths() {
    // CHECK: firrtl.connect %x, %{{[^ ]*}} : !firrtl.uint, !firrtl.uint<4>
    firrtl.instance bar interesting_name @Bar()
    %x = firrtl.wire interesting_name : !firrtl.uint
    %invalid_ui1 = firrtl.invalidvalue : !firrtl.uint
    firrtl.connect %x, %invalid_ui1 : !firrtl.uint, !firrtl.uint
  }
}
