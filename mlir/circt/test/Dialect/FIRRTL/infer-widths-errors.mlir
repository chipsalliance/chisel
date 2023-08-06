// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-infer-widths))' --verify-diagnostics --split-input-file %s

firrtl.circuit "Foo" {
  firrtl.module @Foo(in %clk: !firrtl.clock) {
    // expected-error @+1 {{'firrtl.reg' op is constrained to be wider than itself}}
    %0 = firrtl.reg %clk : !firrtl.clock, !firrtl.uint
    // expected-note @+1 {{constrained width W >= W+3 here}}
    %1 = firrtl.shl %0, 3 : (!firrtl.uint) -> !firrtl.uint
    // expected-note @+1 {{constrained width W >= W+4 here}}
    %2 = firrtl.shl %1, 1 : (!firrtl.uint) -> !firrtl.uint
    // expected-note @+1 {{constrained width W >= 2W+4 here}}
    %3 = firrtl.mul %0, %2 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    // expected-note @+1 {{constrained width W >= 2W+4 here}}
    firrtl.connect %0, %3 : !firrtl.uint, !firrtl.uint
  }
}

// -----
firrtl.circuit "Foo" {
  // expected-note @+1 {{Module `Bar` defined here:}}
  firrtl.extmodule @Bar(in in: !firrtl.uint, 
    out out: !firrtl.bundle<a : uint, b: uint<4>, c: vector<bundle<c: uint>,4>>)
  firrtl.module @Foo(in %in: !firrtl.uint<42>, out %out: !firrtl.bundle<a : uint, b: uint<4>, c: vector<bundle<c: uint>,4>>) {
    // expected-error @+6 {{extern module `Bar` has ports of uninferred width}}
    // expected-note @+5 {{Port: "in"}}
    // expected-note @+4 {{Port: "out"}}
    // expected-note @+3 {{Field: "out.a"}}
    // expected-note @+2 {{Field: "out.c[].c"}}
    // expected-note @+1 {{Only non-extern FIRRTL modules may contain unspecified widths to be inferred automatically.}}
    %inst_in, %inst_out = firrtl.instance inst @Bar(in in: !firrtl.uint, out out: !firrtl.bundle<a : uint, b: uint<4>, c: vector<bundle<c: uint>,4>>)
    firrtl.connect %inst_in, %in : !firrtl.uint, !firrtl.uint<42>
    firrtl.connect %out, %inst_out : !firrtl.bundle<a : uint, b: uint<4>, c: vector<bundle<c: uint>,4>>, !firrtl.bundle<a : uint, b: uint<4>, c: vector<bundle<c: uint>,4>>
  }
}

// -----
firrtl.circuit "Issue1255" {
  // CHECK-LABEL: @Issue1255
  firrtl.module @Issue1255() {
    %tmp74 = firrtl.wire  : !firrtl.uint
    %c14972_ui = firrtl.constant 14972 : !firrtl.uint
    // expected-error @+1 {{amount must be less than or equal operand width}}
    %0 = firrtl.tail %c14972_ui, 15 : (!firrtl.uint) -> !firrtl.uint
    firrtl.connect %tmp74, %0 : !firrtl.uint, !firrtl.uint
  }
}

// -----
firrtl.circuit "Foo" {
  // expected-error @+1 {{uninferred width: port "a" is unconstrained}}
  firrtl.module @Foo (in %a: !firrtl.uint) {
  }
}

// -----
firrtl.circuit "Foo" {
  firrtl.module @Foo () {
    // expected-error @+1 {{uninferred width: wire is unconstrained}}
    %0 = firrtl.wire : !firrtl.uint
  }
}

// -----
firrtl.circuit "Foo" {
  firrtl.module @Foo () {
    // expected-error @+1 {{uninferred width: wire field "[0]" is unconstrained}}
    %0 = firrtl.wire : !firrtl.vector<uint, 16>
  }
}

// -----
firrtl.circuit "Foo" {
  firrtl.module @Foo () {
    // expected-error @+1 {{uninferred width: wire field "a" is unconstrained}}
    %0 = firrtl.wire : !firrtl.bundle<a: uint>
  }
}

// -----
firrtl.circuit "Foo" {
  firrtl.module @Foo () {
    // expected-error @+1 {{uninferred width: wire field "a.b.c" is unconstrained}}
    %0 = firrtl.wire : !firrtl.bundle<a: bundle<b: bundle<c flip: uint, d: uint<1>>>>
  }
}

// -----
// Only first error on module ports reported.
firrtl.circuit "Foo" {
  // expected-error @+2 {{uninferred width: port "a" is unconstrained}}
  // expected-error @+1 {{uninferred width: port "b" is unconstrained}}
  firrtl.module @Foo (in %a: !firrtl.uint, in %b: !firrtl.sint) {
  }
}

// -----
firrtl.circuit "Foo" {
  // expected-error @+2 {{uninferred width: port "a" is unconstrained}}
  // expected-error @+1 {{uninferred width: port "b" width cannot be determined}}
  firrtl.module @Foo(in %a: !firrtl.uint, out %b: !firrtl.uint) {
    // expected-note @+1 {{width is constrained by an uninferred width here:}}
    firrtl.connect %b, %a : !firrtl.uint, !firrtl.uint
  }
}

// -----
firrtl.circuit "Foo"  {
  firrtl.module @Foo() {
    // This should complain about the wire, not the invalid value.
    %0 = firrtl.invalidvalue : !firrtl.bundle<x: uint>
    // expected-error @+1 {{uninferred width: wire "w.x" is unconstrained}}
    %w = firrtl.wire  : !firrtl.bundle<x: uint>
    firrtl.connect %w, %0 : !firrtl.bundle<x: uint>, !firrtl.bundle<x: uint>
  }
}

// -----
// Unsatisfiable widths through AttachOp on analog types should error.
// https://github.com/llvm/circt/issues/4786
firrtl.circuit "AnalogWidths" {
  // expected-error @below {{uninferred width: port "a" cannot satisfy all width requirements}}
  firrtl.module @AnalogWidths(in %a: !firrtl.analog, out %b: !firrtl.analog<2>, out %c: !firrtl.analog<1>) {
    firrtl.attach %a, %b : !firrtl.analog, !firrtl.analog<2>
    // expected-note @below {{width is constrained to be at least 2 here:}}
    // expected-note @below {{width is constrained to be at most 1 here:}}
    firrtl.attach %a, %c : !firrtl.analog, !firrtl.analog<1>
  }
}

// -----
// https://github.com/llvm/circt/issues/4863
firrtl.circuit "Foo" {
  // expected-error @below {{uninferred width: port "out.a" is unconstrained}}
  firrtl.module @Foo(out %out : !firrtl.bundle<a: uint>) {
    %invalid = firrtl.invalidvalue : !firrtl.bundle<a: uint>
    firrtl.connect %out, %invalid : !firrtl.bundle<a: uint>, !firrtl.bundle<a: uint>
  }
}

// -----
// https://github.com/llvm/circt/issues/4863
firrtl.circuit "Foo" {
  // expected-error @below {{uninferred width: port "out" is unconstrained}}
  firrtl.module @Foo(out %out : !firrtl.uint) {
    %invalid = firrtl.invalidvalue : !firrtl.bundle<a: uint>
    %0 = firrtl.subfield %invalid[a] : !firrtl.bundle<a: uint>
    firrtl.connect %out, %0 : !firrtl.uint, !firrtl.uint
  }
}

// -----
// https://github.com/llvm/circt/issues/5002

firrtl.circuit "Issue5002" {
  // expected-error @below {{uninferred width: port "ref" cannot satisfy all width requirements}}
  firrtl.module private @InRef(in %ref : !firrtl.rwprobe<uint>) { }
  firrtl.module @Issue5002(in %x : !firrtl.uint<1>, in %y : !firrtl.uint<2>) {
    %w1, %w1_ref = firrtl.wire forceable : !firrtl.uint, !firrtl.rwprobe<uint>
    %w2, %w2_ref = firrtl.wire forceable : !firrtl.uint, !firrtl.rwprobe<uint>
    firrtl.connect %w1, %x : !firrtl.uint, !firrtl.uint<1>
    firrtl.connect %w2, %y : !firrtl.uint, !firrtl.uint<2>

    %inst1_ref = firrtl.instance inst1 @InRef(in ref: !firrtl.rwprobe<uint>)
    %inst2_ref = firrtl.instance inst2 @InRef(in ref: !firrtl.rwprobe<uint>)
    firrtl.ref.define %inst1_ref, %w1_ref : !firrtl.rwprobe<uint>
    // expected-note @below {{width is constrained to be at most 1 here:}}
    // expected-note @below {{width is constrained to be at least 2 here:}}
    firrtl.ref.define %inst2_ref, %w2_ref : !firrtl.rwprobe<uint>
  }
}

// -----
// https://github.com/llvm/circt/issues/5324

firrtl.circuit "NoWidthEnum" {
  // expected-error @below {{uninferred width: port "o.Some" is unconstrained}}
  firrtl.module @NoWidthEnum(out %o: !firrtl.enum<Some: uint>) {
  }
}
