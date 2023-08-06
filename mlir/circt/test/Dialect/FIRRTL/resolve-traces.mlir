// RUN: circt-opt %s -firrtl-resolve-traces -split-input-file | FileCheck %s

firrtl.circuit "Foo" {
  firrtl.module @Foo() attributes {annotations = [
    {
      class = "chisel3.experimental.Trace$TraceAnnotation",
      chiselTarget = "~Foo|Original"
    }
  ]} {}
}

// Test that a local module annotation is resolved.
//
// CHECK:               class{{.*}}:{{.*}}chisel3.experimental.Trace$TraceAnnotation
// CHECK-SAME{LITERAL}:   \22target\22: \22~Foo|{{0}}\22
// CHECK-SAME:            \22chiselTarget\22:  \22~Foo|Original\22

// Test that the output file is set to include the circuit name.
//
// CHECK-SAME: #hw.output_file<{{.*}}Foo.anno.json

// Test that the symbols are correct.
// CHECK-SAME: symbols = [@Foo]

// -----

firrtl.circuit "Foo" {
  hw.hierpath @path [@Foo::@bar, @Bar]
  firrtl.module @Bar() attributes {annotations = [
    {
      class = "chisel3.experimental.Trace$TraceAnnotation",
      chiselTarget = "~Foo|Original",
      circt.nonlocal = @path
    }
  ]} {}
  firrtl.module @Foo() {
    firrtl.instance bar sym @bar @Bar()
  }
}

// Test that a non-local module annotation is resolved.
//
// CHECK{LITERAL}: ~Foo|{{0}}/{{1}}:{{2}}
// CHECK-SAME:     ~Foo|Original
// CHECK-SAME:     symbols = [@Foo, #hw.innerNameRef<@Foo::@bar>, @Bar]

// -----

firrtl.circuit "Foo" {
  firrtl.module @Foo(
    in %a: !firrtl.uint<1> [
      {
        class = "chisel3.experimental.Trace$TraceAnnotation",
        chiselTarget = "~Foo|Foo>original"
      }
    ]
  ) {}
}

// Test that a local port annotation is resolved.
//
// CHECK{LITERAL}: ~Foo|{{0}}>{{1}}
// CHECK-SAME:     ~Foo|Foo>original
// CHECK-SAME:     symbols = [@Foo, #hw.innerNameRef<@Foo::@[[a_sym:[a-zA-Z0-9_]+]]>]

// Test that the port receives a symbol.
//
// CHECK: in %a: !firrtl.uint<1> sym @[[a_sym]]

// -----

firrtl.circuit "Foo" {
  firrtl.module @Foo() {
    %a = firrtl.wire {annotations = [
      {
        class = "chisel3.experimental.Trace$TraceAnnotation",
        chiselTarget = "~Foo|Foo>original"
      }
    ]} : !firrtl.uint<1>
  }
}

// Test that a local wire annotation is resolved.
//
// CHECK{LITERAL}: ~Foo|{{0}}>{{1}}
// CHECK-SAME:     ~Foo|Foo>original
// CHECK-SAME:     symbols = [@Foo, #hw.innerNameRef<@Foo::@[[a_sym:[a-zA-Z0-9_]+]]>]

// Test that the wire receives a symbol.
//
// CHECK: %a = firrtl.wire sym @[[a_sym]]

// -----

firrtl.circuit "Forceable" {
  firrtl.module @Forceable() {
    %w, %w_ref = firrtl.wire forceable {annotations = [
      {
        class = "chisel3.experimental.Trace$TraceAnnotation",
        chiselTarget = "~Forceable|Forceable>forced"
      }
    ]} : !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>
  }
}

// Test that a local wire annotation is resolved.
//
// CHECK:          sv.verbatim
// CHECK{LITERAL}: ~Forceable|{{0}}>{{1}}
// CHECK-SAME:     ~Forceable|Forceable>forced
// CHECK-SAME:     symbols = [@Forceable, #hw.innerNameRef<@Forceable::@[[w_sym:[a-zA-Z0-9_]+]]>]

// Test that the wire receives a symbol.
//
// CHECK: firrtl.module @Forceable
// CHECK:   %w, %w_ref = firrtl.wire sym @[[w_sym]]

// -----

firrtl.circuit "Foo" {
  firrtl.module @Foo() {
    %a = firrtl.wire {annotations = [
      {
        class = "chisel3.experimental.Trace$TraceAnnotation",
        chiselTarget = "~Foo|Foo>0",
        circt.fieldID = 0
      },
      {
        class = "chisel3.experimental.Trace$TraceAnnotation",
        chiselTarget = "~Foo|Foo>4",
        circt.fieldID = 4
      },
      {
        class = "chisel3.experimental.Trace$TraceAnnotation",
        chiselTarget = "~Foo|Foo>5",
        circt.fieldID = 6
      }
    ]} : !firrtl.vector<bundle<a: uint<1>, b: uint<1>>, 2>
  }
}

// Test that a local wire annotation on an aggregate is resolved for different
// values of field ID.
//
// CHECK{LITERAL}:      ~Foo|{{0}}>{{1}}
// CHECK-SAME:            ~Foo|Foo>0
// CHECK-SAME{LITERAL}: ~Foo|{{0}}>{{1}}[1]
// CHECK-SAME:            ~Foo|Foo>4
// CHECK-SAME{LITERAL}: ~Foo|{{0}}>{{1}}[1].b
// CHECK-SAME:            ~Foo|Foo>5
// CHECK-SAME:          symbols = [@Foo, #hw.innerNameRef<@Foo::@[[a_sym:[a-zA-Z0-9_]+]]>]
//
// CHECK: %a = firrtl.wire sym @[[a_sym]]
