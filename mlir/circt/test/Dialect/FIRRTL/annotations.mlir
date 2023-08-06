// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-annotations))' --split-input-file %s | FileCheck %s

// circt.test copies the annotation to the target
// circt.testNT puts the targetless annotation on the circuit

// Annotations targeting the circuit work.
//
// CHECK-LABEL: firrtl.circuit "Foo"
// CHECK-SAME:    annotations =
// CHECK-SAME:      {class = "circt.testNT", data = "NoTarget"}
// CHECK-SAME:      {class = "circt.test", data = "Target"}
// CHECK-SAME:      {class = "circt.test", data = "CircuitName"}
// CHECK-NOT:     rawAnnotations
firrtl.circuit "Foo" attributes {rawAnnotations = [
  {
    class = "circt.testNT",
    data = "NoTarget"
  },
  {
    class = "circt.test",
    data = "Target",
    target = "~Foo"
  },
  {
    class = "circt.test",
    data = "CircuitName",
    target = "Foo"
  }
]} {
  firrtl.module @Foo() {}
}

// -----

// Annotations targeting modules or external modules work.
//
// CHECK-LABEL: firrtl.circuit "Foo"
// CHECK-NOT:     rawAnnotations
firrtl.circuit "Foo" attributes {rawAnnotations = [
  {
    class = "circt.test",
    data = "Target",
    target = "~Foo|Foo"
  },
  {
    class = "circt.test",
    data = "ModuleName",
    target = "Foo.Foo"
  },
    {
    class = "circt.test",
    data = "ExtModule Target",
    target = "~Foo|Blackbox"
  }
]} {
  // CHECK:      firrtl.module @Foo
  // CHECK-SAME:   annotations =
  // CHECK-SAME:     {class = "circt.test", data = "Target"}
  // CHECK-SAME:     {class = "circt.test", data = "ModuleName"}
  firrtl.module @Foo() {}
  // CHECK:      firrtl.extmodule @Blackbox
  // CHECK-SAME:   annotations =
  // CHECK-SAME:     {class = "circt.test", data = "ExtModule Target"}
  firrtl.extmodule @Blackbox()
}

// -----

// Annotations targeting instances should create NLAs on the module.
//
// CHECK-LABEL: firrtl.circuit "Foo"
// CHECK-NOT:     rawAnnotations
// CHECK-NEXT:    hw.hierpath private @[[nla:[^ ]+]] [@Foo::@[[bar_sym:[^ ]+]], @Bar]
firrtl.circuit "Foo" attributes {rawAnnotations = [
  {
    class = "circt.test",
    data = "a",
    target = "~Foo|Foo>bar"
  },
  {
    class = "circt.test",
    data = "b",
    target = "Foo.Foo.bar"
  },
  {
    class = "circt.test",
    data = "c",
    target = "~Foo|Foo/bar:Bar"
  }
]} {
  // CHECK-NEXT: firrtl.module @Bar()
  // CHECK-SAME:   annotations =
  // CHECK-SAME:     {circt.nonlocal = @[[nla]], class = "circt.test", data = "a"}
  // CHECK-SAME:     {circt.nonlocal = @[[nla]], class = "circt.test", data = "b"}
  // CHECK-SAME:     {circt.nonlocal = @[[nla]], class = "circt.test", data = "c"}
  firrtl.module @Bar() {}
  // CHECK: firrtl.module @Foo
  firrtl.module @Foo() {
    // CHECK-NEXT: firrtl.instance bar sym @[[bar_sym]]
    firrtl.instance bar @Bar()
  }
}

// -----

// Test result annotations of InstanceOp.
//
// Must add inner_sym, if any subfield of a bundle type has nonlocal anchor.
// Otherwise, the nla will be illegal, without any inner_sym.
// Test on port and wire.
//
// CHECK-LABEL: firrtl.circuit "Foo"
// CHECK-NOT:     rawAnnotations
// CHECK-NEXT:    hw.hierpath private @[[nla:[^ ]+]] [@Foo::@[[bar_sym:[^ ]+]], @Bar]
firrtl.circuit "Foo" attributes {rawAnnotations = [
  {
    class = "circt.test",
    data = 0,
    target = "~Foo|Foo>bar.a"
  },
  {
    class = "circt.test",
    data = 1,
    target = "~Foo|Foo>bar.b.baz"
  },
  {
    class = "circt.test",
    data = 2,
    target = "~Foo|Foo/bar:Bar>b.qux"
  },
  {
    class = "circt.test",
    data = 3,
    target = "~Foo|Foo/bar:Bar>d.qux"
  },
  {
    class = "circt.test",
    data = 4,
    target = "Foo.Foo.bar.c"
  }
]} {
  // CHECK-NEXT: firrtl.module @Bar
  // CHECK-SAME:   in %a
  // CHECK-SAME:     {circt.nonlocal = @[[nla]], class = "circt.test", data = 0 : i64}
  // CHECK-SAME:   out %b
  // CHECK-SAME:     {circt.fieldID = 1 : i32, circt.nonlocal = @[[nla]], class = "circt.test", data = 1 : i64}
  // CHECK-SAME:     {circt.fieldID = 2 : i32, circt.nonlocal = @[[nla]], class = "circt.test", data = 2 : i64}
  // CHECK-SAME:   out %c
  // CHECK-SAME:     {circt.nonlocal = @[[nla]], class = "circt.test", data = 4 : i64}
  firrtl.module @Bar(
    in %a: !firrtl.uint<1>,
    out %b: !firrtl.bundle<baz: uint<1>, qux: uint<1>>,
    out %c: !firrtl.uint<1>
  ) {
    // CHECK-NEXT: %d = firrtl.wire
    // CHECK-NOT:    sym
    // CHECK-SAME:   {circt.fieldID = 2 : i32, circt.nonlocal = @[[nla]], class = "circt.test", data = 3 : i64}
    %d = firrtl.wire : !firrtl.bundle<baz: uint<1>, qux: uint<1>>
  }
  // CHECK: firrtl.module @Foo
  firrtl.module @Foo() {
    // CHECK-NEXT: firrtl.instance bar sym @[[bar_sym]]
    %bar_a, %bar_b, %bar_c = firrtl.instance bar @Bar(
      in a: !firrtl.uint<1>,
      out b: !firrtl.bundle<baz: uint<1>, qux: uint<1>>,
      out c: !firrtl.uint<1>
    )
  }
}

// -----

// A ReferenceTarget/ComponentName pointing at a Foo should work.
//
// CHECK-LABEL: firrtl.circuit "Foo"
// CHECK-NOT:     rawAnnotations
firrtl.circuit "Foo"  attributes {rawAnnotations = [
  {
    class = "circt.test",
    data = "a",
    target = "~Foo|Foo>bar"
  },
  {
    class = "circt.test",
    data = "b",
    target = "Foo.Foo.bar"
  }
]} {
  // CHECK: firrtl.module @Foo
  firrtl.module @Foo() {
    // CHECK-NEXT: chirrtl.combmem
    // CHECK-SAME:   {class = "circt.test", data = "a"}
    // CHECK-SAME:   {class = "circt.test", data = "b"}
    %bar = chirrtl.combmem : !chirrtl.cmemory<uint<1>, 8>
  }
}

// -----

// A ReferenceTarget/ComponentName pointing at a memory should work.
//
// CHECK-LABEL: firrtl.circuit "Foo"
// CHECK-NOT:     rawAnnotations
firrtl.circuit "Foo" attributes {rawAnnotations = [
  {
    class = "circt.test",
    data = "a",
    target = "~Foo|Foo>bar"
  },
  {
    class = "circt.test",
    data = "b",
    target = "Foo.Foo.bar"
  }
]} {
  // CHECK: firrtl.module @Foo
  firrtl.module @Foo() {
    // CHECK-NEXT: firrtl.mem
    // CHECK-SAME:   {class = "circt.test", data = "a"}
    // CHECK-SAME:   {class = "circt.test", data = "b"}
    %bar_r = firrtl.mem Undefined {
       depth = 16 : i64,
       name = "bar",
       portNames = ["r"],
       readLatency = 0 : i32,
       writeLatency = 1 : i32
     } : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
  }
}

// -----

// Test result annotations of MemOp.
//
// CHECK-LABEL: firrtl.circuit "Foo"
// CHECK-NOT:     rawAnnotations
firrtl.circuit "Foo" attributes {rawAnnotations = [
  {
    class = "circt.test",
    data = "a",
    target = "~Foo|Foo>bar.r"
  }
  ,
  {
    class = "circt.test",
    data = "b",
    target = "~Foo|Foo>bar.r.data.baz"
  }
  ,
  {
    class = "circt.test",
    data = "c",
    target = "~Foo|Foo>bar.w.en"
  }
  ,
  {
    class = "circt.test",
    data = "d",
    target = "~Foo|Foo>bar.w.data.qux"
  }
]} {
  // CHECK: firrtl.module @Foo
  firrtl.module @Foo() {
    // CHECK-NEXT: firrtl.mem
    // CHECK-SAME:   portAnnotations =
    // CHECK-SAME:     [{class = "circt.test", data = "a"}, {circt.fieldID = 5 : i32, class = "circt.test", data = "b"}]
    // CHECK-SAME:     [{circt.fieldID = 2 : i32, class = "circt.test", data = "c"}, {circt.fieldID = 6 : i32, class = "circt.test", data = "d"}]
    %bar_r, %bar_w = firrtl.mem interesting_name Undefined {
      depth = 16 : i64,
      name = "bar",
      portNames = ["r", "w"],
      readLatency = 0 : i32,
      writeLatency = 1 : i32
    } : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<baz: uint<8>, qux: uint<8>>>,
        !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<baz: uint<8>, qux: uint<8>>, mask: bundle<baz: uint<1>, qux: uint<1>>>
  }
}

// -----

// A ReferenceTarget/ComponentName pointing at a node should work.  This
// shouldn't crash if the node is in a nested block.
//
// CHECK-LABEL: firrtl.circuit "Foo"
firrtl.circuit "Foo" attributes {rawAnnotations = [
  {
    class = "circt.test",
    data = "a",
    target = "~Foo|Foo>bar"
  },
  {
    class = "circt.test",
    data = "b",
    target = "Foo.Foo.baz"
  }
]} {
  firrtl.module @Foo(
    in %clock: !firrtl.clock,
    in %cond_0: !firrtl.uint<1>,
    in %cond_1: !firrtl.uint<1>
  ) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %bar = firrtl.node %c0_ui1  : !firrtl.uint<1>
    firrtl.when %cond_0 : !firrtl.uint<1> {
      firrtl.when %cond_1 : !firrtl.uint<1> {
        %baz = firrtl.node %c0_ui1  : !firrtl.uint<1>
      }
    }
  }
}

// CHECK:      firrtl.module @Foo
// CHECK:        %bar = firrtl.node
// CHECK-SAME:     annotations = [{class = "circt.test", data = "a"}
// CHECK:        %baz = firrtl.node
// CHECK-SAME:     annotations = [{class = "circt.test", data = "b"}]

// -----

// A ReferenceTarget/ComponentName pointing at a wire should work.
//
// CHECK-LABEL: firrtl.circuit "Foo"
firrtl.circuit "Foo" attributes {rawAnnotations = [
  {
    class = "circt.test",
    data = "a",
    target = "~Foo|Foo>bar"
  },
  {
    class = "circt.test",
    data = "b",
    target = "Foo.Foo.bar"
  }
]} {
  firrtl.module @Foo() {
    %bar = firrtl.wire : !firrtl.uint<1>
  }
}

// CHECK:      %bar = firrtl.wire
// CHECK-SAME:   annotations = [{class = "circt.test", data = "a"}, {class = "circt.test", data = "b"}]

// -----

// A ReferenceTarget/ComponentName pointing at a register should work.
//
// CHECK-LABEL: firrtl.circuit "Foo"
firrtl.circuit "Foo" attributes {rawAnnotations = [
  {
    class = "circt.test",
    data = "a",
    target = "~Foo|Foo>bar"
  },
  {
    class = "circt.test",
    data = "b",
    target = "Foo.Foo.baz"
  }
]} {
  firrtl.module @Foo(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>) {
    %bar = firrtl.reg %clock  : !firrtl.clock, !firrtl.uint<1>
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %baz = firrtl.regreset %clock, %reset, %c0_ui1  : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// CHECK:      %bar = firrtl.reg
// CHECK-SAME:   annotations = [{class = "circt.test", data = "a"}]
// CHECK:      %baz = firrtl.regreset
// CHECK-SAME:   annotations = [{class = "circt.test", data = "b"}]

// -----

// A ReferenceTarget/ComponentName pointing at an SeqMem should work.
//
// CHECK-LABEL: firrtl.circuit "Foo"
firrtl.circuit "Foo" attributes {rawAnnotations = [
  {
    class = "circt.test",
    data = "a",
    target = "~Foo|Foo>bar"
  },
  {
    class = "circt.test",
    data = "b",
    target = "Foo.Foo.bar"
  }
]} {
  firrtl.module @Foo() {
    %bar = chirrtl.seqmem Undefined : !chirrtl.cmemory<uint<1>, 8>
  }
}

// CHECK:      chirrtl.seqmem
// CHECK-SAME:   annotations = [{class = "circt.test", data = "a"}, {class = "circt.test", data = "b"}]

// -----

// Subfield/Subindex annotations should be parsed correctly on wires
//
// CHECK-LABEL: firrtl.circuit "Foo"
firrtl.circuit "Foo" attributes {rawAnnotations = [
  {
    class = "circt.test",
    data = "one",
    target = "~Foo|Foo>bar[0]"
  },
  {
    class = "circt.test",
    data = "two",
    target = "~Foo|Foo>bar[1].baz"
  }
]} {
  firrtl.module @Foo() {
    %bar = firrtl.wire : !firrtl.vector<bundle<baz: uint<1>, qux: uint<1>>, 2>
  }
}

// CHECK:      %bar = firrtl.wire {annotations =
// CHECK-SAME:   {circt.fieldID = 1 : i32, class = "circt.test", data = "one"}
// CHECK-SAME:   {circt.fieldID = 5 : i32, class = "circt.test", data = "two"}

// -----

// Subfield/Subindex annotations should be parsed correctly on registers
//
// CHECK-LABEL: firrtl.circuit "Foo"
firrtl.circuit "Foo" attributes {rawAnnotations = [
  {
    class = "circt.test",
    data = "one",
    target = "~Foo|Foo>bar[0]"
  },
  {
    class = "circt.test",
    target = "~Foo|Foo>bar[1].baz",
    data = "two"
  }
]} {
  firrtl.module @Foo(in %clock: !firrtl.clock) {
    %bar = firrtl.reg %clock : !firrtl.clock, !firrtl.vector<bundle<baz: uint<1>, qux: uint<1>>, 2>
  }
}

// CHECK:      %bar = firrtl.reg %clock {annotations =
// CHECK-SAME:   {circt.fieldID = 1 : i32, class = "circt.test", data = "one"}
// CHECK-SAME:   {circt.fieldID = 5 : i32, class = "circt.test", data = "two"}

// -----

// Subindices should not get sign-extended and cause problems.  This circuit has
// caused bugs in the past.
//
// CHECK-LABEL: firrtl.circuit "Foo"
firrtl.circuit "Foo" attributes {rawAnnotations = [
  {
    class = "circt.test",
    data = "a",
    target = "~Foo|Foo>w[9]"
  }
]} {
  firrtl.module @Foo() {
    %w = firrtl.wire  : !firrtl.vector<uint<1>, 18>
  }
}

// CHECK:      %w = firrtl.wire {annotations =
// CHECK-SAME:   {circt.fieldID = 10 : i32, class = "circt.test", data = "a"}

// -----

// A ReferenceTarget/ComponentName pointing at a module/extmodule port should
// work.
//
// CHECK-LABEL: firrtl.circuit "Foo"
firrtl.circuit "Foo" attributes {rawAnnotations = [
  {
    class = "circt.test",
    data = "a",
    target = "~Foo|Bar>bar"
  },
  {
    class = "circt.test",
    data = "b",
    target = "Foo.Foo.foo"
  }
]} {
  firrtl.extmodule @Bar(in bar: !firrtl.uint<1>)
  firrtl.module @Foo(in %foo: !firrtl.uint<1>) {
    %bar_bar = firrtl.instance bar  @Bar(in bar: !firrtl.uint<1>)
    firrtl.strictconnect %bar_bar, %foo : !firrtl.uint<1>
  }
}

// CHECK:      firrtl.extmodule @Bar
// CHECK-SAME:   [[_:.+]] [{class = "circt.test", data = "a"}]
// CHECK:      firrtl.module @Foo
// CHECK-SAME:   %foo: [[_:.+]] [{class = "circt.test", data = "b"}]

// -----

// A module with an instance in its body which has the same name as the module
// itself should not cause issues attaching annotations.
// https://github.com/llvm/circt/issues/2709
firrtl.circuit "Foo" attributes {rawAnnotations = [
  {
    class = "circt.test",
    target = "~Foo|Foo/Foo:Example"
  }
]} {
  firrtl.module @Example() {}
  firrtl.module @Foo() {
    firrtl.instance Foo @Example()
  }
}

// CHECK-LABEL:  firrtl.circuit "Foo"
// CHECK:          hw.hierpath private @[[nla:[^ ]+]] [@Foo::@[[FOO_SYM:.+]], @Example]
// CHECK:          firrtl.module @Example() attributes {
// CHECK-SAME:       annotations = [{circt.nonlocal = @[[nla]], class = "circt.test"}]
// CHECK:          firrtl.module @Foo()
// CHECK:            firrtl.instance Foo sym @[[FOO_SYM]] @Example()

// -----

// Multiple non-local Annotations are supported.
firrtl.circuit "Foo" attributes {rawAnnotations = [
  {class = "circt.test", data = "a", target = "~Foo|Foo/bar:Bar/baz:Baz"},
  {class = "circt.test", data = "b", target = "~Foo|Foo/bar:Bar/baz:Baz"}
]} {
  firrtl.module @Baz() {}
  firrtl.module @Bar() {
    firrtl.instance baz @Baz()
  }
  firrtl.module @Foo() {
    firrtl.instance bar @Bar()
  }
}
// CHECK-LABEL: firrtl.circuit "Foo"
// CHECK:         hw.hierpath private @[[nla:[^ ]+]] [@Foo::@[[BAR_SYM:.+]], @Bar::@[[BAZ_SYM:.+]], @Baz]
// CHECK:         firrtl.module @Baz
// CHECK-SAME:      annotations = [{circt.nonlocal = @[[nla]], class = "circt.test", data = "a"}, {circt.nonlocal = @[[nla]], class = "circt.test", data = "b"}]
// CHECK:         firrtl.module @Bar()
// CHECK:           firrtl.instance baz sym @[[BAZ_SYM]] @Baz()
// CHECK:           firrtl.module @Foo()
// CHECK:           firrtl.instance bar sym @[[BAR_SYM]] @Bar()

// -----

firrtl.circuit "memportAnno"  attributes {rawAnnotations = [
  {
    class = "circt.test",
    target = "~memportAnno|memportAnno/foo:Foo>memory.w"
  }
]} {
  firrtl.module @memportAnno() {
    firrtl.instance foo @Foo()
  }
  firrtl.module @Foo() {
    %memory_w = firrtl.mem Undefined {
      depth = 16 : i64,
      name = "memory",
      portNames = ["w"],
      readLatency = 1 : i32,
      writeLatency = 1 : i32
    } : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
  }
}

// CHECK-LABEL: firrtl.circuit "memportAnno"  {
// CHECK:        hw.hierpath private @nla [@memportAnno::@[[FOO_SYM:.+]], @Foo]
// CHECK: firrtl.instance foo sym @[[FOO_SYM]] @Foo
// CHECK:        %memory_w = firrtl.mem Undefined {depth = 16 : i64, name = "memory", portAnnotations
// CHECK-SAME:   [{circt.nonlocal = @nla, class = "circt.test"}]

// -----

// Test annotation targeting an instance port
// https://github.com/llvm/circt/issues/3340
firrtl.circuit "instportAnno" attributes {rawAnnotations = [
  {
    class = "circt.test",
    target = "~instportAnno|instportAnno/bar:Bar>baz.a"
  }
]} {
  firrtl.module @Baz(out %a: !firrtl.uint<1>) {
    %invalid_ui1 = firrtl.invalidvalue : !firrtl.uint<1>
    firrtl.strictconnect %a, %invalid_ui1 : !firrtl.uint<1>
  }
  firrtl.module @Bar() {
    %baz_a = firrtl.instance baz @Baz(out a: !firrtl.uint<1>)
  }
  firrtl.module @instportAnno() {
    firrtl.instance bar @Bar()
  }
}

// CHECK-LABEL: firrtl.circuit "instportAnno"
// CHECK:        hw.hierpath private @[[HIER:[^ ]+]] [@instportAnno::@[[BAR_SYM:.+]], @Bar::@[[BAZ_SYM:.+]], @Baz]
// CHECK:        firrtl.module @Baz
// CHECK-SAME:     {circt.nonlocal = @[[HIER]], class = "circt.test"}
// CHECK: firrtl.instance baz sym @[[BAZ_SYM]] @Baz
// CHECK: firrtl.instance bar sym @[[BAR_SYM]] @Bar

// -----

// CHECK-LABEL: firrtl.circuit "Aggregates"
firrtl.circuit "Aggregates" attributes {rawAnnotations = [
  {class = "circt.test", target = "~Aggregates|Aggregates>vector[1][1][1]"},
  {class = "circt.test", target = "~Aggregates|Aggregates>bundle.a.b.c"}
  ]} {
  firrtl.module @Aggregates() {
    // CHECK: {annotations = [{circt.fieldID = 14 : i32, class = "circt.test"}]}
    %vector = firrtl.wire  : !firrtl.vector<vector<vector<uint<1>, 2>, 2>, 2>
    // CHECK: {annotations = [{circt.fieldID = 3 : i32, class = "circt.test"}]}
    %bundle = firrtl.wire  : !firrtl.bundle<a: bundle<b: bundle<c: uint<1>>>>
  }
}

// -----

// A non-local annotation should work.

// CHECK-LABEL: firrtl.circuit "FooNL"
// CHECK: hw.hierpath private @nla [@FooNL::@baz, @BazNL::@bar, @BarNL]
// CHECK: firrtl.module @BarNL
// CHECK: %w = firrtl.wire sym @w {annotations = [{circt.nonlocal = @nla, class = "circt.test", nl = "nl"}]}
// CHECK: %w2 = firrtl.wire sym @w2 {annotations = [{circt.fieldID = 5 : i32, circt.nonlocal = @nla, class = "circt.test", nl = "nl2"}]} : !firrtl.bundle<a: uint, b: vector<uint, 4>>
// CHECK: firrtl.instance bar sym @bar @BarNL()
// CHECK: firrtl.instance baz sym @baz @BazNL()
// CHECK: firrtl.module @FooL
// CHECK: %w3 = firrtl.wire {annotations = [{class = "circt.test", nl = "nl3"}]}
firrtl.circuit "FooNL"  attributes {rawAnnotations = [
  {class = "circt.test", nl = "nl", target = "~FooNL|FooNL/baz:BazNL/bar:BarNL"},
  {class = "circt.test", nl = "nl", target = "~FooNL|FooNL/baz:BazNL/bar:BarNL>w"},
  {class = "circt.test", nl = "nl2", target = "~FooNL|FooNL/baz:BazNL/bar:BarNL>w2.b[2]"},
  {class = "circt.test", nl = "nl3", target = "~FooNL|FooL>w3"}
  ]}  {
  firrtl.module @BarNL() {
    %w = firrtl.wire  sym @w : !firrtl.uint
    %w2 = firrtl.wire sym @w2 : !firrtl.bundle<a: uint, b: vector<uint, 4>>
    firrtl.skip
  }
  firrtl.module @BazNL() {
    firrtl.instance bar sym @bar @BarNL()
  }
  firrtl.module @FooNL() {
    firrtl.instance baz sym @baz @BazNL()
  }
  firrtl.module @FooL() {
    %w3 = firrtl.wire: !firrtl.uint
  }
}

// -----

// Non-local annotations on memory ports should work.

// CHECK-LABEL: firrtl.circuit "MemPortsNL"
// CHECK: hw.hierpath private @nla [@MemPortsNL::@[[CHILD_SYM:.+]], @Child]
// CHECK: firrtl.module @Child()
// CHECK:   %bar_r = firrtl.mem
// CHECK-NOT: sym
// CHECK-SAME: portAnnotations = {{\[}}[{circt.nonlocal = @nla, class = "circt.test", nl = "nl"}]]
// CHECK: firrtl.module @MemPortsNL()
// CHECK:   firrtl.instance child sym @[[CHILD_SYM]]
firrtl.circuit "MemPortsNL" attributes {rawAnnotations = [
  {class = "circt.test", nl = "nl", target = "~MemPortsNL|MemPortsNL/child:Child>bar.r"}
  ]}  {
  firrtl.module @Child() {
    %bar_r = firrtl.mem Undefined  {depth = 16 : i64, name = "bar", portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
  }
  firrtl.module @MemPortsNL() {
    firrtl.instance child @Child()
  }
}

// -----

// Annotations on ports should work.
firrtl.circuit "Test" attributes {rawAnnotations = [
  {class = "circt.test", target = "~Test|PortTest>in"}
  ]} {
  firrtl.module @PortTest(in %in : !firrtl.uint<1>) {}
  firrtl.module @Test() {
    %portttest_in = firrtl.instance porttest @PortTest(in in : !firrtl.uint<1>)
  }
}

// -----

// Subannotations on ports should work.
firrtl.circuit "Test" attributes {rawAnnotations = [
  {class = "circt.test", target = "~Test|PortTest>in.a"}
  ]} {
  // CHECK: firrtl.module @PortTest(in %in: !firrtl.bundle<a: uint<1>> [{circt.fieldID = 1 : i32, class = "circt.test"}])
  firrtl.module @PortTest(in %in : !firrtl.bundle<a: uint<1>>) {}
  firrtl.module @Test() {
    %portttest_in = firrtl.instance porttest @PortTest(in in : !firrtl.bundle<a: uint<1>>)
  }
}
// -----

// Annotations on instances should be moved to the target module.
firrtl.circuit "Test" attributes {rawAnnotations = [
  {class = "circt.test", target = "~Test|Test>exttest"}
  ]} {
  // CHECK: hw.hierpath private @nla [@Test::@[[EXTTEST:.+]], @ExtTest]
  // CHECK: firrtl.extmodule @ExtTest() attributes {annotations = [{circt.nonlocal = @nla, class = "circt.test"}]}
  firrtl.extmodule @ExtTest()

  firrtl.module @Test() {
    // CHECK: firrtl.instance exttest sym @[[EXTTEST]] @ExtTest()
    firrtl.instance exttest @ExtTest()
  }
}

// -----

// Annotations on instances should be moved to the target module.
firrtl.circuit "Test" attributes {rawAnnotations = [
  {class = "circt.test", target = "~Test|Test>exttest.in"}
  ]} {
  // CHECK: hw.hierpath private @nla [@Test::@[[EXTTEST:.+]], @ExtTest]
  // CHECK: firrtl.extmodule @ExtTest(in in: !firrtl.uint<1> [{circt.nonlocal = @nla, class = "circt.test"}])
  firrtl.extmodule @ExtTest(in in: !firrtl.uint<1>)

  firrtl.module @Test() {
    // CHECK: %exttest_in = firrtl.instance exttest sym @[[EXTTEST]] @ExtTest(in in: !firrtl.uint<1>)
    firrtl.instance exttest @ExtTest(in in : !firrtl.uint<1>)
  }
}

// -----

firrtl.circuit "Test" attributes {rawAnnotations =[
  {class = "circt.ConventionAnnotation", target = "~Test|Test", convention = "scalarized"}
  ]} {
  // CHECK: attributes {convention = #firrtl<convention scalarized>}
  firrtl.module @Test() attributes {convention = #firrtl<convention internal>} {}
}

// -----

// DontTouchAnnotations are placed on the things they target.

firrtl.circuit "Foo"  attributes {
  rawAnnotations = [
    {class = "firrtl.transforms.DontTouchAnnotation", target = "~Foo|Foo>_T_0"},
    {class = "firrtl.transforms.DontTouchAnnotation", target = "~Foo|Foo>_T_1"},
    {class = "firrtl.transforms.DontTouchAnnotation", target = "~Foo|Foo>_T_2"},
    {class = "firrtl.transforms.DontTouchAnnotation", target = "~Foo|Foo>_T_3"},
    {class = "firrtl.transforms.DontTouchAnnotation", target = "~Foo|Foo>_T_4"},
    {class = "firrtl.transforms.DontTouchAnnotation", target = "~Foo|Foo>_T_5"},
    {class = "firrtl.transforms.DontTouchAnnotation", target = "~Foo|Foo>_T_6"},
    {class = "firrtl.transforms.DontTouchAnnotation", target = "~Foo|Foo>_T_8"},
    {class = "firrtl.transforms.DontTouchAnnotation", target = "~Foo|Foo>_T_9.a"},
    {class = "firrtl.transforms.DontTouchAnnotation", target = "~Foo|Foo>_T_10.a"},
    {class = "firrtl.transforms.DontTouchAnnotation", target = "~Foo|Foo/bar:Bar>_T.a"}]} {
  // CHECK:      hw.hierpath private @nla [@Foo::@[[BAR_SYM:.+]], @Bar]
  // CHECK-NEXT: firrtl.module @Foo
  firrtl.module @Foo(in %reset: !firrtl.uint<1>, in %clock: !firrtl.clock) {
    // CHECK-NEXT: %_T_0 = firrtl.wire {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]}
    %_T_0 = firrtl.wire  : !firrtl.uint<1>
    // CHECK-NEXT: %_T_1 = firrtl.node %_T_0 {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]}
    %_T_1 = firrtl.node %_T_0  : !firrtl.uint<1>
    // CHECK-NEXT: %_T_2 = firrtl.reg %clock {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]}
    %_T_2 = firrtl.reg %clock  : !firrtl.clock, !firrtl.uint<1>
    %c0_ui4 = firrtl.constant 0 : !firrtl.uint<4>
    // CHECK: %_T_3 = firrtl.regreset
    // CHECK-SAME: {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]}
    %_T_3 = firrtl.regreset %clock, %reset, %c0_ui4  : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<4>
    // CHECK-NEXT: %_T_4 = chirrtl.seqmem
    // CHECK-SAME: {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]}
    %_T_4 = chirrtl.seqmem Undefined  : !chirrtl.cmemory<vector<uint<1>, 9>, 256>
    // CHECK-NEXT: %_T_5 = chirrtl.combmem {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]}
    %_T_5 = chirrtl.combmem  : !chirrtl.cmemory<vector<uint<1>, 9>, 256>
    // CHECK: chirrtl.memoryport Infer %_T_5 {annotations =
    // CHECK-SAME: {class = "firrtl.transforms.DontTouchAnnotation"}
    %_T_6_data, %_T_6_port = chirrtl.memoryport Infer %_T_5  {name = "_T_6"} : (!chirrtl.cmemory<vector<uint<1>, 9>, 256>) -> (!firrtl.vector<uint<1>, 9>, !chirrtl.cmemoryport)
    chirrtl.memoryport.access %_T_6_port[%reset], %clock : !chirrtl.cmemoryport, !firrtl.uint<1>, !firrtl.clock
    // CHECK: firrtl.mem
    // CHECK-SAME: {class = "firrtl.transforms.DontTouchAnnotation"}
    %_T_8_w = firrtl.mem Undefined  {depth = 8 : i64, name = "_T_8", portNames = ["w"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: uint<4>, mask: uint<1>>
    %aggregate = firrtl.wire  : !firrtl.bundle<a: uint<1>>
    // CHECK: %_T_9 = firrtl.node %aggregate {annotations = [{circt.fieldID = 1 : i32, class = "firrtl.transforms.DontTouchAnnotation"}]}
    %_T_9 = firrtl.node %aggregate  : !firrtl.bundle<a: uint<1>>
    // CHECK: instance bar sym @[[BAR_SYM]] @Bar()
    firrtl.instance bar @Bar()

    // CHECK: %_T_10, %_T_10_ref = firrtl.node %aggregate forceable {annotations = [{circt.fieldID = 1 : i32, class = "firrtl.transforms.DontTouchAnnotation"}]}
    %_T_10, %_T_10_ref = firrtl.node %aggregate forceable : !firrtl.bundle<a: uint<1>>
  }
  firrtl.module @Bar() {
    //  CHECK: %_T = firrtl.wire {annotations = [{circt.fieldID = 1 : i32, circt.nonlocal = @nla, class = "firrtl.transforms.DontTouchAnnotation"}]}
    %_T = firrtl.wire : !firrtl.bundle<a: uint<1>>
  }
}

// -----

firrtl.circuit "GCTInterface"  attributes {
  annotations = [
    {unrelatedAnnotation}
  ],
  rawAnnotations = [
    {
      class = "sifive.enterprise.grandcentral.GrandCentralView$SerializedViewAnnotation",
      companion = "~GCTInterface|view_companion",
      name = "view",
      parent = "~GCTInterface|GCTInterface",
      view =
        {
          class = "sifive.enterprise.grandcentral.AugmentedBundleType",
          defName = "ViewName",
          elements = [
            {
              description = "the register in GCTInterface",
              name = "register",
              tpe =
                {
                  class = "sifive.enterprise.grandcentral.AugmentedBundleType",
                  defName = "Register",
                  elements = [
                    {
                      name = "_2",
                      tpe =
                        {
                          class = "sifive.enterprise.grandcentral.AugmentedVectorType",
                          elements = [
                            {
                              class = "sifive.enterprise.grandcentral.AugmentedGroundType",
                              ref =
                                {
                                  circuit = "GCTInterface",
                                  component = [
                                    {
                                      class = "firrtl.annotations.TargetToken$Field",
                                      value = "_2"
                                    },
                                    {
                                      class = "firrtl.annotations.TargetToken$Index",
                                      value = 0 : i64
                                    }
                                  ],
                                  module = "GCTInterface",
                                  path = [],
                                  ref = "r"
                                },
                              tpe =
                                {
                                  class = "sifive.enterprise.grandcentral.GrandCentralView$UnknownGroundType$"
                                }
                            },
                            {
                              class = "sifive.enterprise.grandcentral.AugmentedGroundType",
                              ref =
                                {
                                  circuit = "GCTInterface",
                                  component = [
                                    {
                                      class = "firrtl.annotations.TargetToken$Field",
                                      value = "_2"
                                    },
                                    {
                                      class = "firrtl.annotations.TargetToken$Index",
                                      value = 1 : i64
                                    }
                                  ],
                                  module = "GCTInterface",
                                  path = [],
                                  ref = "r"
                                },
                              tpe =
                                {
                                  class = "sifive.enterprise.grandcentral.GrandCentralView$UnknownGroundType$"
                                }
                            }
                          ]
                        }
                    },
                    {
                      name = "_0_inst",
                      tpe =
                        {
                          class = "sifive.enterprise.grandcentral.AugmentedBundleType",
                          defName = "_0_def",
                          elements = [
                            {
                              name = "_1",
                              tpe =
                                {
                                  class = "sifive.enterprise.grandcentral.AugmentedGroundType",
                                  ref =
                                    {
                                      circuit = "GCTInterface",
                                      component = [
                                        {
                                          class = "firrtl.annotations.TargetToken$Field",
                                          value = "_0"
                                        },
                                        {
                                          class = "firrtl.annotations.TargetToken$Field",
                                          value = "_1"
                                        }
                                      ],
                                      module = "GCTInterface",
                                      path = [],
                                      ref = "r"
                                    },
                                  tpe =
                                    {
                                      class = "sifive.enterprise.grandcentral.GrandCentralView$UnknownGroundType$"
                                    }
                                }
                            },
                            {
                              name = "_0",
                              tpe =
                                {
                                  class = "sifive.enterprise.grandcentral.AugmentedGroundType",
                                  ref =
                                    {
                                      circuit = "GCTInterface",
                                      component = [
                                        {
                                          class = "firrtl.annotations.TargetToken$Field",
                                          value = "_0"
                                        },
                                        {
                                          class = "firrtl.annotations.TargetToken$Field",
                                          value = "_0"
                                        }
                                      ],
                                      module = "GCTInterface",
                                      path = [],
                                      ref = "r"
                                    },
                                  tpe =
                                    {
                                      class = "sifive.enterprise.grandcentral.GrandCentralView$UnknownGroundType$"
                                    }
                                }
                            }
                          ]
                        }
                    }
                  ]
                }
            },
            {
              description = "element of the register 'forceable_reg' in GCTInterface",
              name = "forceable_reg_element",
              tpe =
                {
                  class = "sifive.enterprise.grandcentral.AugmentedGroundType",
                  ref =
                    {
                      circuit = "GCTInterface",
                      component = [
                        {
                          class = "firrtl.annotations.TargetToken$Field",
                          value = "_2"
                        },
                        {
                          class = "firrtl.annotations.TargetToken$Index",
                          value = 1 : i64
                        }
                      ],
                      module = "GCTInterface",
                      path = [],
                      ref = "forceable_reg"
                    },
                  tpe =
                    {
                      class = "sifive.enterprise.grandcentral.GrandCentralView$UnknownGroundType$"
                    }
                }
            },
            {
              description = "the port 'a' in GCTInterface",
              name = "port",
              tpe =
                {
                  class = "sifive.enterprise.grandcentral.AugmentedGroundType",
                  ref =
                    {
                      circuit = "GCTInterface",
                      component = [],
                      module = "GCTInterface",
                      path = [],
                      ref = "a"
                    },
                  tpe =
                    {
                      class = "sifive.enterprise.grandcentral.GrandCentralView$UnknownGroundType$"
                    }
                }
            }
          ]
        }
      }
  ]
} {
  firrtl.module private @view_companion() {
    firrtl.skip
  }
  firrtl.module @GCTInterface(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %a: !firrtl.uint<1>) {
    %r = firrtl.reg %clock : !firrtl.clock, !firrtl.bundle<_0: bundle<_0: uint<1>, _1: uint<1>>, _2: vector<uint<1>, 2>>
    %forceable_reg, %forceable_reg_ref = firrtl.reg %clock forceable : !firrtl.clock, !firrtl.bundle<_0: bundle<_0: uint<1>, _1: uint<1>>, _2: vector<uint<1>, 2>>, !firrtl.rwprobe<bundle<_0: bundle<_0: uint<1>, _1: uint<1>>, _2: vector<uint<1>, 2>>>
    firrtl.instance view_companion  @view_companion()
  }
}

// CHECK-LABEL: firrtl.circuit "GCTInterface"

// The interface definition should show up as a circuit annotation.  Nested
// interfaces show up as nested bundle types and not as separate interfaces.
// CHECK-SAME: annotations
// CHECK-SAME: {unrelatedAnnotation}
// CHECK-SAME: {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
// CHECK-SAME:  defName = "ViewName",
// CHECK-SAME:  elements = [
// CHECK-SAME:    {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
// CHECK-SAME:     defName = "Register",
// CHECK-SAME:     description = "the register in GCTInterface",
// CHECK-SAME:     elements = [
// CHECK-SAME:       {class = "sifive.enterprise.grandcentral.AugmentedVectorType",
// CHECK-SAME:        elements = [
// CHECK-SAME:          {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
// CHECK-SAME:           id = [[ID_2_0:[0-9]+]] : i64,
// CHECK-SAME:           name = "_2"},
// CHECK-SAME:          {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
// CHECK-SAME:           id = [[ID_2_1:[0-9]+]] : i64,
// CHECK-SAME:           name = "_2"}],
// CHECK-SAME:        name = "_2"},
// CHECK-SAME:       {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
// CHECK-SAME:        defName = "_0_def",
// CHECK-SAME:        elements = [
// CHECK-SAME:          {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
// CHECK-SAME:           id = [[ID_1:[0-9]+]] : i64,
// CHECK-SAME:           name = "_1"},
// CHECK-SAME:          {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
// CHECK-SAME:           id = [[ID_0:[0-9]+]] : i64,
// CHECK-SAME:           name = "_0"}],
// CHECK-SAME:        name = "_0_inst"}],
// CHECK-SAME:     name = "register"},
// CHECK-SAME:    {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
// CHECK-SAME:     description = "element of the register 'forceable_reg' in GCTInterface",
// CHECK-SAME:     id = [[ID_forceable_reg:[0-9]+]] : i64,
// CHECK-SAME:     name = "forceable_reg_element"},
// CHECK-SAME:    {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
// CHECK-SAME:     description = "the port 'a' in GCTInterface",
// CHECK-SAME:     id = [[ID_port:[0-9]+]] : i64,
// CHECK-SAME:     name = "port"}],
// CHECK-SAME:  id = [[ID_ViewName:[0-9]+]] : i64,
// CHECK-SAME:  name = "view"}

// The companion should be marked.
// CHECK:      firrtl.module private @view_companion(
// CHECK-SAME:   in %[[port_0:[a-zA-Z0-9_]+]]: !firrtl.uint<1>,
// CHECK-SAME:   in %[[port_1:[a-zA-Z0-9_]+]]: !firrtl.uint<1>,
// CHECK-SAME:   in %[[port_2:[a-zA-Z0-9_]+]]: !firrtl.uint<1>,
// CHECK-SAME:   in %[[port_3:[a-zA-Z0-9_]+]]: !firrtl.uint<1>,
// CHECK-SAME:   in %[[port_4:[a-zA-Z0-9_]+]]: !firrtl.uint<1>
// CHECK-SAME: )
// CHECK-SAME: annotations
// CHECK-SAME: {class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
// CHECK-SAME:  id = [[ID_ViewName]] : i64,
// CHECK-SAME:  name = "view"}
//
// CHECK:      firrtl.node %[[port_0]]
// CHECK-SAME:   {class = "sifive.enterprise.grandcentral.AugmentedGroundType", id = 1 : i64}
// CHECK:      firrtl.node %[[port_1]]
// CHECK-SAME:   {class = "sifive.enterprise.grandcentral.AugmentedGroundType", id = 2 : i64}
// CHECK:      firrtl.node %[[port_2]]
// CHECK-SAME:   {class = "sifive.enterprise.grandcentral.AugmentedGroundType", id = 3 : i64}
// CHECK:      firrtl.node %[[port_3]]
// CHECK-SAME:   {class = "sifive.enterprise.grandcentral.AugmentedGroundType", id = 4 : i64}
// CHECK:      firrtl.node %[[port_4]]
// CHECK-SAME:   {class = "sifive.enterprise.grandcentral.AugmentedGroundType", id = 5 : i64}

// The RefSend must be generated.
// CHECK: firrtl.module @GCTInterface
// CHECK-SAME: %a: !firrtl.uint<1>
// CHECK:      %r = firrtl.reg %clock : !firrtl.clock, !firrtl.bundle<_0: bundle<_0: uint<1>, _1: uint<1>>, _2: vector<uint<1>, 2>>
// CHECK:      %forceable_reg, %forceable_reg_ref = firrtl.reg %clock forceable : !firrtl.clock, !firrtl.bundle<_0: bundle<_0: uint<1>, _1: uint<1>>, _2: vector<uint<1>, 2>>, !firrtl.rwprobe<bundle<_0: bundle<_0: uint<1>, _1: uint<1>>, _2: vector<uint<1>, 2>>>
// CHECK:      %[[view_companion_view__2refPort:.+]], %[[view_companion_view__2refPort_1:.+]], %[[view_companion_view__1refPort:.+]], %[[view_companion_view__0refPort:.+]], %[[view_companion_view_portrefPort:.+]] = firrtl.instance view_companion  @view_companion(in {{.*}}: !firrtl.uint<1>, in {{.*}}: !firrtl.uint<1>, in {{.*}}: !firrtl.uint<1>, in {{.*}}: !firrtl.uint<1>, in {{.*}}: !firrtl.uint<1>)
// CHECK:      %0 = firrtl.subfield %r[_2] : !firrtl.bundle<_0: bundle<_0: uint<1>, _1: uint<1>>, _2: vector<uint<1>, 2>>
// CHECK:      %1 = firrtl.subindex %0[0] : !firrtl.vector<uint<1>, 2>
// CHECK:      %2 = firrtl.subfield %r[_2] : !firrtl.bundle<_0: bundle<_0: uint<1>, _1: uint<1>>, _2: vector<uint<1>, 2>>
// CHECK:      %3 = firrtl.subindex %2[1] : !firrtl.vector<uint<1>, 2>
// CHECK:      %4 = firrtl.subfield %r[_0] : !firrtl.bundle<_0: bundle<_0: uint<1>, _1: uint<1>>, _2: vector<uint<1>, 2>>
// CHECK:      %5 = firrtl.subfield %4[_1] : !firrtl.bundle<_0: uint<1>, _1: uint<1>>
// CHECK:      %6 = firrtl.subfield %r[_0] : !firrtl.bundle<_0: bundle<_0: uint<1>, _1: uint<1>>, _2: vector<uint<1>, 2>>
// CHECK:      %7 = firrtl.subfield %6[_0] : !firrtl.bundle<_0: uint<1>, _1: uint<1>>
// CHECK:      %8 = firrtl.subfield %forceable_reg[_2] : !firrtl.bundle<_0: bundle<_0: uint<1>, _1: uint<1>>, _2: vector<uint<1>, 2>>
// CHECK:      %9 = firrtl.subindex %8[1] : !firrtl.vector<uint<1>, 2>
// CHECK:      firrtl.strictconnect %view_companion_view_register__2_0__bore, %1 : !firrtl.uint<1>
// CHECK:      firrtl.strictconnect %view_companion_view_register__2_1__bore, %3 : !firrtl.uint<1>
// CHECK:      firrtl.strictconnect %view_companion_view_register__0_inst__1__bore, %5 : !firrtl.uint<1>
// CHECK:      firrtl.strictconnect %view_companion_view_register__0_inst__0__bore, %7 : !firrtl.uint<1>
// CHECK:      firrtl.strictconnect %view_companion_view_forceable_reg_element__bore, %9 : !firrtl.uint<1>
// CHECK:      firrtl.strictconnect %view_companion_view_port__bore, %a : !firrtl.uint<1>

// -----

firrtl.circuit "Foo"  attributes {rawAnnotations = [
  {
    class = "sifive.enterprise.grandcentral.ViewAnnotation",
    companion = "~Foo|Bar_companion",
    name = "Bar",
    parent = "~Foo|Foo",
    view =
      {
        class = "sifive.enterprise.grandcentral.AugmentedBundleType",
        defName = "View",
        elements = [
          {
            description = "a string",
            name = "string",
            tpe =
              {
                class = "sifive.enterprise.grandcentral.AugmentedStringType",
                value = "hello"
              }
          },
          {
            description = "a boolean",
            name = "boolean",
            tpe =
              {
                class = "sifive.enterprise.grandcentral.AugmentedBooleanType",
                value = false
              }
          },
          {
            description = "an integer",
            name = "integer",
            tpe =
              {
                class = "sifive.enterprise.grandcentral.AugmentedIntegerType",
                value = 42 : i64
              }
          },
          {
            description = "a double",
            name = "double",
            tpe =
              {
                class = "sifive.enterprise.grandcentral.AugmentedDoubleType",
                value = 3.140000e+00 : f64
              }
          }
        ]
      }
  }
]} {
  firrtl.module private @Bar_companion() {
    firrtl.skip
  }
  firrtl.module @Foo() {
     firrtl.instance Bar_companion @Bar_companion()
   }
}

// CHECK-LABEL: firrtl.circuit "Foo"
// CHECK-SAME: annotations = [{class = "[[_:.+]]AugmentedBundleType", [[_:.+]] elements = [{
// CHECK-SAME: "sifive.enterprise.grandcentral.AugmentedStringType"
// CHECK-SAME: "sifive.enterprise.grandcentral.AugmentedBooleanType"
// CHECK-SAME: "sifive.enterprise.grandcentral.AugmentedIntegerType"
// CHECK-SAME: "sifive.enterprise.grandcentral.AugmentedDoubleType"

// -----

// SiFive-custom annotations related to the GrandCentral utility.  These
// annotations do not conform to standard SingleTarget or NoTarget format and
// need to be manually split up.

// Test sifive.enterprise.grandcentral.DataTapsAnnotation with all possible
// variants of DataTapKeys.

// SiFive-custom annotations related to the GrandCentral utility.  These
// annotations do not conform to standard SingleTarget or NoTarget format and
// need to be manually split up.

// Test sifive.enterprise.grandcentral.DataTapsAnnotation with all possible
// variants of DataTapKeys.

firrtl.circuit "GCTDataTap" attributes {rawAnnotations = [{
  class = "sifive.enterprise.grandcentral.DataTapsAnnotation",
  keys = [
    {
      class = "sifive.enterprise.grandcentral.ReferenceDataTapKey",
      sink = "~GCTDataTap|GCTDataTap>tap_0",
      source = "~GCTDataTap|GCTDataTap>r"
    },
    {
      class = "sifive.enterprise.grandcentral.ReferenceDataTapKey",
      sink = "~GCTDataTap|GCTDataTap>tap_1[0]",
      source = "~GCTDataTap|GCTDataTap>r"
    },
    {
      class = "sifive.enterprise.grandcentral.ReferenceDataTapKey",
      sink = "~GCTDataTap|GCTDataTap>tap_2",
      source = "~GCTDataTap|GCTDataTap>w.a"
    },
    {
      class = "sifive.enterprise.grandcentral.ReferenceDataTapKey",
      sink = "~GCTDataTap|GCTDataTap>tap_3[0]",
      source = "~GCTDataTap|GCTDataTap>w.a"
    },
    {
      class = "sifive.enterprise.grandcentral.DataTapModuleSignalKey",
      internalPath = "baz.qux",
      module = "~GCTDataTap|BlackBox",
      sink = "~GCTDataTap|GCTDataTap>tap_4"
    },
    {
      class = "sifive.enterprise.grandcentral.DataTapModuleSignalKey",
      internalPath = "baz.quz",
      module = "~GCTDataTap|BlackBox",
      sink = "~GCTDataTap|GCTDataTap>tap_5[0]"
    },
    {
      class = "sifive.enterprise.grandcentral.ReferenceDataTapKey",
      sink = "~GCTDataTap|GCTDataTap>tap_6",
      source = "~GCTDataTap|GCTDataTap/im:InnerMod>w"
    }
  ]
}]} {
  firrtl.extmodule private @BlackBox() attributes {defname = "BlackBox"}
  firrtl.module private @InnerMod() {
    %w = firrtl.wire : !firrtl.uint<1>
  }
  firrtl.module @GCTDataTap(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>) {
    %r = firrtl.reg %clock  : !firrtl.clock, !firrtl.uint<1>
    %w = firrtl.wire : !firrtl.bundle<a: uint<1>>
    %tap_0 = firrtl.wire : !firrtl.uint<1>
    %tap_1 = firrtl.wire : !firrtl.vector<uint<1>, 1>
    %tap_2 = firrtl.wire : !firrtl.uint<1>
    %tap_3 = firrtl.wire : !firrtl.vector<uint<1>, 1>
    %tap_4 = firrtl.wire : !firrtl.uint<1>
    %tap_5 = firrtl.wire : !firrtl.vector<uint<1>, 1>
    %tap_6 = firrtl.wire : !firrtl.uint<1>

    firrtl.instance BlackBox @BlackBox()
    firrtl.instance im @InnerMod()
  }
}

// CHECK-LABEL: firrtl.extmodule private @BlackBox
// CHECK-SAME:    out {{[a-zA-Z0-9_]+}}: !firrtl.probe<uint<1>>
// CHECK-SAME:    out {{[a-zA-Z0-9_]+}}: !firrtl.probe<uint<1>>
// CHECK-SAME:    internalPaths = ["baz.qux", "baz.quz"]

// CHECK-LABEL: firrtl.module private @InnerMod
// CHECK-SAME:    out %[[tap_6:[a-zA-Z0-9_]+]]: !firrtl.probe<uint<1>>
// CHECK:         %[[w_ref:[a-zA-Z09_]+]] = firrtl.ref.send %w
// CHECK:         firrtl.ref.define %[[tap_6]], %[[w_ref]]

// CHECK-LABEL: firrtl.module @GCTDataTap
// CHECK:         %[[w_a0:[a-zA-Z0-9_]+]] = firrtl.subfield %w[a]
// CHECK-NEXT:    %[[w_a1:[a-zA-Z0-9_]+]] = firrtl.subfield %w[a]
//
// CHECK-DAG:    %tap_0 = firrtl.node %r
//
// CHECK-DAG:    %[[tap_1_0:[a-zA-Z0-9_]+]] = firrtl.subindex %tap_1[0]
// CHECK-DAG:    firrtl.strictconnect %[[tap_1_0]], %r
//
// CHECK-DAG:    %tap_2 = firrtl.node %[[w_a1]]
//
// CHECK-DAG:    %[[tap_3_0:[a-zA-Z0-9_]+]] = firrtl.subindex %tap_3[0]
// CHECK-DAG:    firrtl.strictconnect %[[tap_3_0]], %[[w_a0]]
//
// CHECK-DAG:    %[[tap_4_port:[a-zA-Z0-9_]+]], %[[tap_5_port:[a-zA-Z0-9_]+]] = firrtl.instance BlackBox
// CHECK-DAG:    %[[tap_4_resolve:[a-zA-Z0-9_]+]] = firrtl.ref.resolve %[[tap_4_port]]
// CHECK-DAG:    %tap_4 = firrtl.node %[[tap_4_resolve]]
//
// CHECK-DAG:    %[[tap_5_resolve:[a-zA-Z0-9_]+]] = firrtl.ref.resolve %[[tap_5_port]]
// CHECK-DAG:    %[[tap_5_0:[a-zA-Z0-9_]+]] = firrtl.subindex %tap_5[0]
// CHECK-DAG:    firrtl.strictconnect %[[tap_5_0]], %[[tap_5_resolve]]
//
// CHECK-DAG:    %[[tap_6_port:[a-zA-Z0-9_]+]] = firrtl.instance im @InnerMod
// CHECK-DAG:    %[[tap_6_resolve:[a-zA-Z0-9_]+]] = firrtl.ref.resolve %[[tap_6_port]]
// CHECK-DAG:    %tap_6 = firrtl.node %[[tap_6_resolve]]

// -----

// Test sifive.enterprise.grandcentral.MemTapAnnotation
firrtl.circuit "GCTMemTap" attributes {rawAnnotations = [{
  class = "sifive.enterprise.grandcentral.MemTapAnnotation",
  source = "~GCTMemTap|GCTMemTap>mem",
  sink = ["GCTMemTap.GCTMemTap.memTap[0]", "~GCTMemTap|GCTMemTap>mem[1]"]
}]} {
  firrtl.module @GCTMemTap() {
    %mem = chirrtl.combmem  : !chirrtl.cmemory<uint<1>, 2>
    %memTap = firrtl.wire : !firrtl.vector<uint<1>, 2>
  }
}


// CHECK-LABEL: firrtl.circuit "GCTMemTap"

// CHECK:      firrtl.module @GCTMemTap
// CHECK:        %[[debug_port:[a-zA-Z0-9_]+]] = chirrtl.debugport %mem
// CHECK-SAME:     {name = "memTap"}
// CHECK-SAME:     (!chirrtl.cmemory<uint<1>, 2>) -> !firrtl.probe<vector<uint<1>, 2>>
// CHECK-NEXT:   %[[debug_port_resolve:[a-zA-Z0-9_]+]] = firrtl.ref.resolve %[[debug_port]]
// CHECK-NEXT:   %memTap = firrtl.node %[[debug_port_resolve]]

// -----

// CHECK-LABEL: firrtl.circuit "GrandCentralViewsBundle"
firrtl.circuit "GrandCentralViewsBundle"  attributes {
  rawAnnotations = [
    {
      class = "sifive.enterprise.grandcentral.ViewAnnotation",
      companion = "~GrandCentralViewsBundle|Companion",
      name = "View",
      parent = "~GrandCentralViewsBundle|GrandCentralViewsBundle",
      view =
        {
          class = "sifive.enterprise.grandcentral.AugmentedBundleType",
          defName = "MyInterface",
          elements = [
            {
              name = "b",
              tpe = {
                class = "sifive.enterprise.grandcentral.AugmentedBundleType",
                defName = "SubInterface",
                elements = [
                  {
                    name = "a",
                    tpe =
                      {
                        class = "sifive.enterprise.grandcentral.AugmentedGroundType",
                        ref =
                          {
                            circuit = "GrandCentralViewsBundle",
                            component = [
                              {
                                class = "firrtl.annotations.TargetToken$Field",
                                value = "a"
                              }
                            ],
                            module = "GrandCentralViewsBundle",
                            path = [
                              {
                                _1 =
                                  {
                                    class = "firrtl.annotations.TargetToken$Instance",
                                    value = "bar"
                                  },
                                _2 =
                                  {
                                    class = "firrtl.annotations.TargetToken$OfModule",
                                    value = "Bar"
                                  }
                              }
                            ],
                            ref = "a"
                          },
                        tpe =
                          {
                            class = "sifive.enterprise.grandcentral.GrandCentralView$UnknownGroundType$"
                          }
                      }
                  },
                  {
                    name = "b",
                    tpe =
                      {
                        class = "sifive.enterprise.grandcentral.AugmentedGroundType",
                        ref =
                          {
                            circuit = "GrandCentralViewsBundle",
                            component = [
                              {
                                class = "firrtl.annotations.TargetToken$Field",
                                value = "b"
                              }
                            ],
                            module = "GrandCentralViewsBundle",
                            path = [
                              {
                                _1 =
                                  {
                                    class = "firrtl.annotations.TargetToken$Instance",
                                    value = "bar"
                                  },
                                _2 =
                                  {
                                    class = "firrtl.annotations.TargetToken$OfModule",
                                    value = "Bar"
                                  }
                              }
                            ],
                            ref = "a"
                          },
                          tpe =
                            {
                              class = "sifive.enterprise.grandcentral.GrandCentralView$UnknownGroundType$"
                            }
                      }
                  }
                ]
              }
            }
          ]
        }
    }
  ]
} {
  // CHECK:      firrtl.module @Companion
  // CHECK-SAME:   in %[[port_0:[a-zA-Z0-9_]+]]: !firrtl.uint<1>
  // CHECK-SAME:   in %[[port_1:[a-zA-Z0-9_]+]]: !firrtl.uint<2>
  // CHECK-SAME:   {class = "sifive.enterprise.grandcentral.ViewAnnotation.companion", id = 0 : i64, name = "View"}
  firrtl.module @Companion() {
    // CHECK-NEXT: firrtl.node %[[port_0]]
    // CHECK-SAME:   {annotations = [{class = "sifive.enterprise.grandcentral.AugmentedGroundType", id = 1 : i64}]}
    // CHECK-SAME:   !firrtl.uint<1>
    // CHECK-NEXT: firrtl.node %[[port_1]]
    // CHECK-SAME:   {annotations = [{class = "sifive.enterprise.grandcentral.AugmentedGroundType", id = 2 : i64}]}
    // CHECK-SAME:   !firrtl.uint<2>
  }
  // CHECK:      firrtl.module @Bar
  // CHECK-SAME:   out %[[refPort_0:[a-zA-Z0-9_]+]]: !firrtl.probe<uint<1>>
  // CHECK-SAME:   out %[[refPort_1:[a-zA-Z0-9_]+]]: !firrtl.probe<uint<2>>
  firrtl.module @Bar() {
    %a = firrtl.wire interesting_name  : !firrtl.bundle<a: uint<1>, b: uint<2>>
    // CHECK:      %[[a_0:[a-zA-Z0-9_]+]] = firrtl.subfield %a[a]
    // CHECK-NEXT: %[[a_1:[a-zA-Z0-9_]+]] = firrtl.subfield %a[b]
    // CHECK-NEXT: %[[a_0_ref:[a-zA-Z0-9_]+]] = firrtl.ref.send %[[a_0]]
    // CHECK-NEXT: firrtl.ref.define %[[refPort_0]], %[[a_0_ref]]
    // CHECK-NEXT: %[[a_1_ref:[a-zA-Z0-9_]+]] = firrtl.ref.send %[[a_1]]
    // CHECK-NEXT: firrtl.ref.define %[[refPort_1]], %[[a_1_ref]]
  }
  // CHECK:      firrtl.module @GrandCentralViewsBundle()
  firrtl.module @GrandCentralViewsBundle() {
    // CHECK-NEXT: %[[bar_refPort_0:[a-zA-Z0-9_]+]], %[[bar_refPort_1:[a-zA-Z0-9_]+]] = firrtl.instance bar
    firrtl.instance bar @Bar()
    // CHECK-NEXT: %[[companion_port_0:[a-zA-Z0-9_]+]], %[[companion_port_1:[a-zA-Z0-9_]+]] = firrtl.instance companion
    firrtl.instance companion @Companion()
    // CHECK-NEXT: %[[bar_refPort_0_resolve:[a-zA-Z0-9_]+]] = firrtl.ref.resolve %[[bar_refPort_0]]
    // CHECK-NEXT: firrtl.strictconnect %[[companion_port_0]], %[[bar_refPort_0_resolve]]
    // CHECK-NEXT: %[[bar_refPort_1_resolve:[a-zA-Z0-9_]+]] = firrtl.ref.resolve %[[bar_refPort_1]]
    // CHECK-NEXT: firrtl.strictconnect %[[companion_port_1]], %[[bar_refPort_1_resolve]]
  }
}

// -----

firrtl.circuit "Top"  attributes {rawAnnotations = [{
  class = "sifive.enterprise.grandcentral.DataTapsAnnotation",
  keys = [
    {
       class = "sifive.enterprise.grandcentral.ReferenceDataTapKey",
       source = "~Top|Top/foo:Foo/b:Bar>inv", sink = "~Top|Top>tap"
    }
  ]}]} {
  // CHECK-LABEL: firrtl.circuit "Top"  {
  // CHECK-NOT:   "sifive.enterprise.grandcentral.DataTapsAnnotation"
  // CHECK:  firrtl.module private @Bar(out %inv__bore: !firrtl.probe<uint<1>>)
  firrtl.module private @Bar() {
    %inv = firrtl.wire interesting_name  : !firrtl.uint<1>
    // CHECK:  %0 = firrtl.ref.send %inv : !firrtl.uint<1>
    // CHECK:  firrtl.ref.define %inv__bore, %0 : !firrtl.probe<uint<1>>
  }
  // CHECK-LABEL: firrtl.module private @Foo(out %b_inv__bore: !firrtl.probe<uint<1>>)
  firrtl.module private @Foo() {
    firrtl.instance b interesting_name  @Bar()
    // CHECK:  %[[b_inv:[a-zA-Z0-9_]+]] = firrtl.instance b interesting_name  @Bar(out inv__bore: !firrtl.probe<uint<1>>)
    // CHECK:  firrtl.ref.define %b_inv__bore, %[[b_inv]] : !firrtl.probe<uint<1>>
  }
  // CHECK: firrtl.module @Top()
  firrtl.module @Top() {
    firrtl.instance foo interesting_name  @Foo()
    %tap = firrtl.wire interesting_name  : !firrtl.uint<1>
    // CHECK:  %[[foo_b_inv:[a-zA-Z0-9_]+]] = firrtl.instance foo interesting_name  @Foo(out b_inv__bore: !firrtl.probe<uint<1>>)
    // CHECK:  %0 = firrtl.ref.resolve %[[foo_b_inv]] : !firrtl.probe<uint<1>>
    // CHECK:  %tap = firrtl.node %0 : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "Top"  attributes {rawAnnotations = [
  {
    class = "sifive.enterprise.grandcentral.DataTapsAnnotation",
    keys = [
      {
        class = "sifive.enterprise.grandcentral.DataTapModuleSignalKey",
        internalPath = "random.something",
        module = "~Top|Bar",
        sink = "~Top|Top>tap"
      },
      {
        class = "sifive.enterprise.grandcentral.DataTapModuleSignalKey",
        internalPath = "random.something.external",
        module = "~Top|ExtBar",
        sink = "~Top|Top>tap2"
      }
    ]}]} {
  firrtl.extmodule private @ExtBar()
  // CHECK: firrtl.extmodule private @ExtBar(out random_something_external: !firrtl.probe<uint<1>>)
  // CHECK-SAME: internalPaths = ["random.something.external"]
  // CHECK:  firrtl.module private @Bar(out %[[_gen_ref2:.+]]: !firrtl.probe<uint<1>>)
  // CHECK:  %[[random:.+]] = firrtl.verbatim.expr "random.something" : () -> !firrtl.uint<1>
  // CHECK:  %0 = firrtl.ref.send %[[random]] : !firrtl.uint<1>
  // CHECK:  firrtl.ref.define %[[_gen_ref2]], %0 : !firrtl.probe<uint<1>>
  firrtl.module private @Bar() {
  }

  // CHECK-LABEL:  firrtl.module private @Foo(
  // CHECK-SAME: out %b_random_something__bore: !firrtl.probe<uint<1>>, out %b2_random_something_external__bore: !firrtl.probe<uint<1>>
  firrtl.module private @Foo() {
    firrtl.instance b interesting_name  @Bar()
    // CHECK:  %[[gen_refPort:.+]] = firrtl.instance b interesting_name @Bar
    // CHECK-SAME: (out [[_gen_ref2]]: !firrtl.probe<uint<1>>)
    firrtl.instance b2 interesting_name  @ExtBar()
    // CHECK: %b2_random_something_external = firrtl.instance b2 interesting_name  @ExtBar(out random_something_external: !firrtl.probe<uint<1>>)
  }
  // CHECK-LABEL: firrtl.module @Top()
  firrtl.module @Top() {
    firrtl.instance foo interesting_name  @Foo()
    %tap = firrtl.wire interesting_name  : !firrtl.uint<1>
    %tap2 = firrtl.wire interesting_name  : !firrtl.uint<1>
    // CHECK:  %[[foo__gen_tap:.+]], %[[foo__gen_tap2:.+]] = firrtl.instance foo interesting_name  @Foo
    // CHECK-SAME: (out b_random_something__bore: !firrtl.probe<uint<1>>, out b2_random_something_external__bore: !firrtl.probe<uint<1>>)
    // CHECK:  %[[v0:.+]] = firrtl.ref.resolve %[[foo__gen_tap]] : !firrtl.probe<uint<1>>
    // CHECK:  %tap = firrtl.node %[[v0]] : !firrtl.uint<1>
  }
}

// -----
// Test with Parent module not being the LCA.

// CHECK-LABEL: firrtl.circuit "GrandCentralParentIsNotLCA"
firrtl.circuit "GrandCentralParentIsNotLCA"  attributes {
  rawAnnotations = [
    {
      class = "sifive.enterprise.grandcentral.ViewAnnotation",
      companion = "~GrandCentralParentIsNotLCA|Companion",
      name = "View",
      parent = "~GrandCentralParentIsNotLCA|GrandCentralParentIsNotLCA",
      view =
        {
          class = "sifive.enterprise.grandcentral.AugmentedBundleType",
          defName = "MyInterface",
          elements = [
            {
              name = "b",
              tpe = {
                class = "sifive.enterprise.grandcentral.AugmentedGroundType",
                ref =
                  {
                    circuit = "GrandCentralParentIsNotLCA",
                    component = [],
                    module = "GrandCentralParentIsNotLCA",
                    path = [
                      {
                         _1 =
                           {
                             class = "firrtl.annotations.TargetToken$Instance",
                             value = "companion"
                           },
                         _2 =
                           {
                             class = "firrtl.annotations.TargetToken$OfModule",
                             value = "Companion"
                           }
                      },
                      {
                         _1 =
                           {
                             class = "firrtl.annotations.TargetToken$Instance",
                             value = "bar"
                           },
                         _2 =
                           {
                             class = "firrtl.annotations.TargetToken$OfModule",
                             value = "Bar"
                           }
                      }
                    ],
                    ref = "a"
                  },
                  tpe = "sifive.enterprise.grandcentral.GrandCentralView$UnknownGroundType$"
                }
            }
          ]
        }
    }
  ]
} {
  // CHECK:        firrtl.module @Bar
  // CHECK-SAME:     out %[[a_refPort:[a-zA-Z0-9_]+]]: !firrtl.probe<uint<1>>
  firrtl.module @Bar() {
    %a = firrtl.wire : !firrtl.uint<1>
    // CHECK:        %[[a_ref:[a-zA-Z0-9_]+]] = firrtl.ref.send %a
    // CHECK-NEXT:   firrtl.ref.define %[[a_refPort]], %[[a_ref]]
  }
  // CHECK:        firrtl.module @Companion()
  firrtl.module @Companion() {
    firrtl.instance bar @Bar()
    // CHECK-NEXT:   %[[bar_a_refPort:[a-zA-Z0-9_]+]] = firrtl.instance bar
    // CHECK-NEXT:   %[[b_refResolve:[a-zA-Z0-9_]+]] = firrtl.ref.resolve %[[bar_a_refPort]]
    // CHECK-NEXT:   firrtl.node %[[b_refResolve]]
    // CHECK-SAME:     {class = "sifive.enterprise.grandcentral.AugmentedGroundType", id = 1 : i64}
    // CHECK-SAME:     : !firrtl.uint<1>
  }
  firrtl.module @GrandCentralParentIsNotLCA() {
    firrtl.instance companion @Companion()
  }
}

// -----

// Check that Grand Central View Annotations are applied properly when they are
// targeting things inside the companion.  Specifically, this should work for
// both ports and components, e.g., registers.  This does not emit any RefSendOps
// or RefResolveOps and instead just does a direct connection.

// CHECK-LABEL: "GrandCentralViewInsideCompanion"
// CHECK-SAME:    id = [[aId:[0-9]+]] : i64, name = "a"
// CHECK-SAME:    id = [[bId:[0-9]+]] : i64, name = "b"
firrtl.circuit "GrandCentralViewInsideCompanion" attributes {
  rawAnnotations = [
    {
      class = "sifive.enterprise.grandcentral.ViewAnnotation",
      name = "View",
      companion = "~GrandCentralViewInsideCompanion|Companion",
      parent = "~GrandCentralViewInsideCompanion|GrandCentralViewInsideCompanion",
      view = {
        class = "sifive.enterprise.grandcentral.AugmentedBundleType",
        defName = "MyInterface",
        elements = [
          {
            name = "a",
            tpe = {
              class = "sifive.enterprise.grandcentral.AugmentedGroundType",
              ref = {
                circuit = "GrandCentralViewInsideCompanion",
                module = "GrandCentralViewInsideCompanion",
                path = [
                  {
                    _1 = {
                      class = "firrtl.annotations.TargetToken$Instance",
                      value = "companion"
                    },
                    _2 = {
                      class = "firrtl.annotations.TargetToken$OfModule",
                      value = "Companion"
                    }
                  }
                ],
                ref = "a",
                component = []
              },
              tpe = {
                class = "sifive.enterprise.grandcentral.GrandCentralView$UnknownGroundType$"
              }
            }
          },
          {
            name = "b",
            tpe = {
              class = "sifive.enterprise.grandcentral.AugmentedGroundType",
              ref = {
                circuit = "GrandCentralViewInsideCompanion",
                module = "GrandCentralViewInsideCompanion",
                path = [
                  {
                    _1 = {
                      class = "firrtl.annotations.TargetToken$Instance",
                      value = "companion"
                    },
                    _2 = {
                      class = "firrtl.annotations.TargetToken$OfModule",
                      value = "Companion"
                    }
                  }
                ],
                ref = "b",
                component = []
              },
              tpe = {
                class = "sifive.enterprise.grandcentral.GrandCentralView$UnknownGroundType$"
              }
            }
          }
        ]
      }
    }
  ]
} {
  // CHECK:      firrtl.module @Companion
  firrtl.module @Companion(out %b: !firrtl.uint<2>) {
    %clock = firrtl.specialconstant 0 : !firrtl.clock
    %a = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<1>
    // CHECK:      firrtl.node %a
    // CHECK-SAME:   {class = "sifive.enterprise.grandcentral.AugmentedGroundType", id = [[aId]] : i64}
    // CHECK-SAME:   : !firrtl.uint<1>
    // CHECK-NEXT: firrtl.node %b
    // CHECK-SAME:   {class = "sifive.enterprise.grandcentral.AugmentedGroundType", id = [[bId]] : i64}
    // CHECK-SAME:   : !firrtl.uint<2>
  }
  firrtl.module @GrandCentralViewInsideCompanion() {
    %companion_b = firrtl.instance companion @Companion(out b: !firrtl.uint<2>)
  }
}

// -----

// Check that TraceNameAnnotation (which has don't touch semantics) is expanded
// into a TraceAnnotation (which does not have don't touch semantics) and a
// DontTouchAnnotation whenever this targets something that can be a legal
// target of a DontTouchAnnotation.

// CHECK-LABEL: firrtl.circuit "TraceNameAnnotation"
firrtl.circuit "TraceNameAnnotation" attributes {rawAnnotations = [
    {
      class = "chisel3.experimental.Trace$TraceNameAnnotation",
      chiselTarget = "~TraceNameAnnotation|TraceNameAnnotation",
      target = "~TraceNameAnnotation|TraceNameAnnotation"
    },
    {
      class = "chisel3.experimental.Trace$TraceNameAnnotation",
      chiselTarget = "~TraceNameAnnotation|Foo",
      target = "~TraceNameAnnotation|Foo"
    },
    {
      class = "chisel3.experimental.Trace$TraceNameAnnotation",
      chiselTarget = "~TraceNameAnnotation|TraceNameAnnotation/foo:Foo",
      target = "~TraceNameAnnotation|TraceNameAnnotation/foo:Foo"
    },
    {
      class = "chisel3.experimental.Trace$TraceNameAnnotation",
      chiselTarget = "~TraceNameAnnotation|TraceNameAnnotation>w",
      target = "~TraceNameAnnotation|TraceNameAnnotation>w"
    }
  ]} {
  // CHECK:      firrtl.extmodule @Foo()
  // CHECK-SAME:   {chiselTarget = "~TraceNameAnnotation|Foo"
  // CHECK-SAME:    class = "chisel3.experimental.Trace$TraceAnnotation"}
  // CHECK-SAME:   {chiselTarget = "~TraceNameAnnotation|TraceNameAnnotation/foo:Foo"
  // CHECK-SAME:    circt.nonlocal =
  // CHECK-SAME:    class = "chisel3.experimental.Trace$TraceAnnotation"}
  firrtl.extmodule @Foo()
  // CHECK:      firrtl.module @TraceNameAnnotation()
  // CHECK-SAME:   {chiselTarget = "~TraceNameAnnotation|TraceNameAnnotation"
  // CHECK-SAME:    class = "chisel3.experimental.Trace$TraceAnnotation"}
  firrtl.module @TraceNameAnnotation() {
    firrtl.instance foo @Foo()
    // CHECK:      %w = firrtl.wire
    // CHECK-SAME:   {chiselTarget = "~TraceNameAnnotation|TraceNameAnnotation>w"
    // CHECK-SAME:    class = "chisel3.experimental.Trace$TraceAnnotation"}
    // CHECK-SAME:   {class = "firrtl.transforms.DontTouchAnnotation"}
    %w = firrtl.wire : !firrtl.uint<1>
  }
}

// -----

// Test that the valid types are connected, when the source has un-inferred width but sink has width.
firrtl.circuit "Top"  attributes {rawAnnotations = [{
  class = "sifive.enterprise.grandcentral.DataTapsAnnotation",
  keys = [{
    class = "sifive.enterprise.grandcentral.ReferenceDataTapKey",
    sink = "~Top|Top>tap", source = "~Top|Foo>sum"
    }]}]} {
  firrtl.module private @Foo() {
    %sum = firrtl.wire : !firrtl.uint
  }
  // CHECK-LABEL: firrtl.module @Top
  firrtl.module @Top() {
    firrtl.instance foo interesting_name  @Foo()
    %tap = firrtl.wire interesting_name  : !firrtl.uint<8>
    // CHECK:   %[[v0:.+]] = firrtl.ref.resolve %foo_sum__bore : !firrtl.probe<uint>
    // CHECK:   %tap = firrtl.node %[[v0]] : !firrtl.uint
  }
}

// -----

// Test that sub-field of a DataTap sink with internal path is handled correctly.
firrtl.circuit "Top"  attributes {rawAnnotations = [{
  class = "sifive.enterprise.grandcentral.DataTapsAnnotation",
  keys = [{
    class = "sifive.enterprise.grandcentral.DataTapModuleSignalKey",
    internalPath = "random.something",
    module = "~Top|BlackBox",
    sink = "~Top|Top>tap2.wid"
    }]}]} {
  firrtl.extmodule private @BlackBox() attributes {defname = "BlackBox"}
  // CHECK:  firrtl.extmodule private @BlackBox
  // CHECK-SAME:  out [[gen_ref:.+]]: !firrtl.probe<uint<1>>)
  // CHECK-SAME: attributes {defname = "BlackBox", internalPaths = ["random.something"]}
  firrtl.module @Top(in %in: !firrtl.uint<1>) {
    %tap2 = firrtl.wire : !firrtl.bundle<wid: uint<1>>
    firrtl.instance localparam @BlackBox()
    // CHECK:  %[[localparam__gen_ref:.+]] = firrtl.instance localparam @BlackBox(out [[gen_ref]]: !firrtl.probe<uint<1>>)
    // CHECK:  firrtl.ref.resolve %[[localparam__gen_ref]] : !firrtl.probe<uint<1>>
  }
}

// -----

// Test memory initialization setting.
// CHECK-LABEL: firrtl.circuit "MemoryInitializationAnnotations"
firrtl.circuit "MemoryInitializationAnnotations" attributes {
  rawAnnotations = [
    {
      class = "firrtl.annotations.LoadMemoryAnnotation",
      fileName = "mem1.txt",
      hexOrBinary = "b",
      originalMemoryNameOpt = "m",
      target = "~MemoryInitializationAnnotations|MemoryInitializationAnnotations>m1"
    },
    {
      class = "firrtl.annotations.MemoryFileInlineAnnotation",
      filename = "mem2.txt",
      hexOrBinary = "h",
      target = "~MemoryInitializationAnnotations|MemoryInitializationAnnotations>m2"
    },
    {
      class = "firrtl.annotations.LoadMemoryAnnotation",
      fileName = "mem3.txt",
      hexOrBinary = "b",
      originalMemoryNameOpt = "m",
      target = "~MemoryInitializationAnnotations|MemoryInitializationAnnotations>m3"
    },
    {
      class = "firrtl.annotations.MemoryFileInlineAnnotation",
      filename = "mem4.txt",
      hexOrBinary = "h",
      target = "~MemoryInitializationAnnotations|MemoryInitializationAnnotations>m4"
    },
    {
      class = "firrtl.annotations.LoadMemoryAnnotation",
      fileName = "mem5.txt",
      hexOrBinary = "b",
      originalMemoryNameOpt = "m",
      target = "~MemoryInitializationAnnotations|MemoryInitializationAnnotations>m5"
    },
    {
      class = "firrtl.annotations.MemoryFileInlineAnnotation",
      filename = "mem6.txt",
      hexOrBinary = "h",
      target = "~MemoryInitializationAnnotations|MemoryInitializationAnnotations>m6"
    }
  ]
} {
  firrtl.module @MemoryInitializationAnnotations() {
    // CHECK:      %m1_r = firrtl.mem
    // CHECK-SAME:   #firrtl.meminit<"mem1.txt", true, false>
    %m1_r = firrtl.mem Undefined {
      depth = 2 : i64,
      name = "m1",
      portNames = ["r"],
      readLatency = 1 : i32,
      writeLatency = 1 : i32
    } : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<8>>
    // CHECK-NEXT: %m2_r = firrtl.mem
    // CHECK-SAME:   #firrtl.meminit<"mem2.txt", false, true>
    %m2_r = firrtl.mem Undefined {
      depth = 2 : i64,
      name = "m2",
      portNames = ["r"],
      readLatency = 1 : i32,
      writeLatency = 1 : i32
    } : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<8>>
    // CHECK-NEXT: %m3 = chirrtl.seqmem Undefined {init = #firrtl.meminit<"mem3.txt", true, false>}
    %m3 = chirrtl.seqmem Undefined : !chirrtl.cmemory<uint<8>, 32>
    // CHECK-NEXT: %m4 = chirrtl.seqmem Undefined {init = #firrtl.meminit<"mem4.txt", false, true>}
    %m4 = chirrtl.seqmem Undefined : !chirrtl.cmemory<uint<8>, 32>
    // CHECK-NEXT: %m5 = chirrtl.combmem {init = #firrtl.meminit<"mem5.txt", true, false>}
    %m5 = chirrtl.combmem : !chirrtl.cmemory<uint<8>, 32>
    // CHECK-NEXT: %m6 = chirrtl.combmem {init = #firrtl.meminit<"mem6.txt", false, true>}
    %m6 = chirrtl.combmem : !chirrtl.cmemory<uint<8>, 32>
  }
}

// -----

// CHECK-LABEL: firrtl.circuit "Top"
firrtl.circuit "Top" attributes {
  rawAnnotations = [
    {
      class = "sifive.enterprise.firrtl.MarkDUTAnnotation",
      target = "~Top|DUT"
    }]
  } {
  // CHECK-LABEL: firrtl.module
  // CHECK-NOT:     private
  // CHECK-SAME:     @DUT()
  // CHECK-SAME:    class = "sifive.enterprise.firrtl.MarkDUTAnnotation"
  firrtl.module private @DUT() {}

  // CHECK-LABEL:      firrtl.module @Top
  // CHECK-NEXT:   firrtl.instance dut @DUT
  firrtl.module @Top() {
    firrtl.instance dut @DUT()
  }
}
