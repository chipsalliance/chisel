// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-check-comb-loops))' --split-input-file --verify-diagnostics %s | FileCheck %s

// Loop-free circuit
// CHECK: firrtl.circuit "hasnoloops"
firrtl.circuit "hasnoloops"   {
  firrtl.module @thru(in %in1: !firrtl.uint<1>, in %in2: !firrtl.uint<1>, out %out1: !firrtl.uint<1>, out %out2: !firrtl.uint<1>) {
    firrtl.connect %out1, %in1 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %out2, %in2 : !firrtl.uint<1>, !firrtl.uint<1>
  }
  firrtl.module @hasnoloops(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>) {
    %x = firrtl.wire  : !firrtl.uint<1>
    %inner_in1, %inner_in2, %inner_out1, %inner_out2 = firrtl.instance inner @thru(in in1: !firrtl.uint<1>, in in2: !firrtl.uint<1>, out out1: !firrtl.uint<1>, out out2: !firrtl.uint<1>)
    firrtl.connect %inner_in1, %a : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %x, %inner_out1 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %inner_in2, %x : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %b, %inner_out2 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

// Simple combinational loop
// CHECK-NOT: firrtl.circuit "hasloops"
firrtl.circuit "hasloops"   {
  // expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: hasloops.{y <- z <- y}}}
  firrtl.module @hasloops(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
    %y = firrtl.wire  : !firrtl.uint<1>
    %z = firrtl.wire  : !firrtl.uint<1>
    firrtl.connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %z, %y : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %y, %z : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %d, %z : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

// Single-element combinational loop
// CHECK-NOT: firrtl.circuit "loop"
firrtl.circuit "loop"   {
  // expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: loop.{w <- w}}}
  firrtl.module @loop(out %y: !firrtl.uint<8>) {
    %w = firrtl.wire  : !firrtl.uint<8>
    firrtl.connect %w, %w : !firrtl.uint<8>, !firrtl.uint<8>
  }
}

// -----

// Node combinational loop
// CHECK-NOT: firrtl.circuit "hasloops"
firrtl.circuit "hasloops"   {
  // expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: hasloops.{y <- z <- ... <- y}}}
  firrtl.module @hasloops(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
    %y = firrtl.wire  : !firrtl.uint<1>
    firrtl.connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>
    %0 = firrtl.and %c, %y : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    %z = firrtl.node %0  : !firrtl.uint<1>
    firrtl.connect %y, %z : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %d, %z : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

// Combinational loop through a combinational memory read port
// CHECK-NOT: firrtl.circuit "hasloops"
firrtl.circuit "hasloops"   {
  // expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: hasloops.{y <- z <- m.r.data <- m.r.addr <- y}}}
  firrtl.module @hasloops(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
    %y = firrtl.wire  : !firrtl.uint<1>
    %z = firrtl.wire  : !firrtl.uint<1>
    firrtl.connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>
    %m_r = firrtl.mem Undefined  {depth = 2 : i64, name = "m", portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
    %0 = firrtl.subfield %m_r[clk] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
    firrtl.connect %0, %clk : !firrtl.clock, !firrtl.clock
    %1 = firrtl.subfield %m_r[addr] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
    firrtl.connect %1, %y : !firrtl.uint<1>, !firrtl.uint<1>
    %2 = firrtl.subfield %m_r[en] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
    %c1_ui = firrtl.constant 1 : !firrtl.uint
    firrtl.connect %2, %c1_ui : !firrtl.uint<1>, !firrtl.uint
    %3 = firrtl.subfield %m_r[data] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
    firrtl.connect %z, %3 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %y, %z : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %d, %z : !firrtl.uint<1>, !firrtl.uint<1>
  }
}


// -----

// Combination loop through an instance
// CHECK-NOT: firrtl.circuit "hasloops"
firrtl.circuit "hasloops"   {
  firrtl.module @thru(in %in: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
    firrtl.connect %out, %in : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: hasloops.{y <- z <- inner.out <- inner.in <- y}}}
  firrtl.module @hasloops(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
    %y = firrtl.wire  : !firrtl.uint<1>
    %z = firrtl.wire  : !firrtl.uint<1>
    firrtl.connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>
    %inner_in, %inner_out = firrtl.instance inner @thru(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    firrtl.connect %inner_in, %y : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %z, %inner_out : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %y, %z : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %d, %z : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

// Multiple simple loops in one SCC
// CHECK-NOT: firrtl.circuit "hasloops"
firrtl.circuit "hasloops"   {
  // expected-error @below {{hasloops.{c <- b <- ... <- a <- ... <- c}}}
  firrtl.module @hasloops(in %i: !firrtl.uint<1>, out %o: !firrtl.uint<1>) {
    %a = firrtl.wire  : !firrtl.uint<1>
    %b = firrtl.wire  : !firrtl.uint<1>
    %c = firrtl.wire  : !firrtl.uint<1>
    %d = firrtl.wire  : !firrtl.uint<1>
    %e = firrtl.wire  : !firrtl.uint<1>
    %0 = firrtl.and %c, %i : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %a, %0 : !firrtl.uint<1>, !firrtl.uint<1>
    %1 = firrtl.and %a, %d : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %b, %1 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>
    %2 = firrtl.and %c, %e : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %d, %2 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %e, %b : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %o, %e : !firrtl.uint<1>, !firrtl.uint<1>
  }
}


// -----

firrtl.circuit "strictConnectAndConnect" {
  // expected-error @below {{strictConnectAndConnect.{b <- a <- b}}}
  firrtl.module @strictConnectAndConnect(out %a: !firrtl.uint<11>, out %b: !firrtl.uint<11>) {
    %w = firrtl.wire : !firrtl.uint<11>
    firrtl.strictconnect %b, %w : !firrtl.uint<11>
    firrtl.connect %a, %b : !firrtl.uint<11>, !firrtl.uint<11>
    firrtl.strictconnect %b, %a : !firrtl.uint<11>
  }
}

// -----

firrtl.circuit "vectorRegInit"   {
  firrtl.module @vectorRegInit(in %clk: !firrtl.clock) {
    %reg = firrtl.reg %clk : !firrtl.clock, !firrtl.vector<uint<8>, 2>
    %0 = firrtl.subindex %reg[0] : !firrtl.vector<uint<8>, 2>
    firrtl.connect %0, %0 : !firrtl.uint<8>, !firrtl.uint<8>
  }
}

// -----

firrtl.circuit "bundleRegInit"   {
  firrtl.module @bundleRegInit(in %clk: !firrtl.clock) {
    %reg = firrtl.reg %clk : !firrtl.clock, !firrtl.bundle<a: uint<1>>
    %0 = firrtl.subfield %reg[a] : !firrtl.bundle<a: uint<1>>
    firrtl.connect %0, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "PortReadWrite"  {
  firrtl.extmodule private @Bar(in a: !firrtl.uint<1>)
  // expected-error @below {{PortReadWrite.{a <- bar.a <- a}}}
  firrtl.module @PortReadWrite() {
    %a = firrtl.wire : !firrtl.uint<1>
    %bar_a = firrtl.instance bar interesting_name  @Bar(in a: !firrtl.uint<1>)
    firrtl.strictconnect %bar_a, %a : !firrtl.uint<1>
    firrtl.strictconnect %a, %bar_a : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "Foo"  {
  firrtl.module private @Bar(in %a: !firrtl.uint<1>) {}
  // expected-error @below {{Foo.{bar.a <- a <- bar.a}}}
  firrtl.module @Foo(out %a: !firrtl.uint<1>) {
    %bar_a = firrtl.instance bar interesting_name  @Bar(in a: !firrtl.uint<1>)
    firrtl.strictconnect %bar_a, %a : !firrtl.uint<1>
    firrtl.strictconnect %a, %bar_a : !firrtl.uint<1>
  }
}

// -----

// Node combinational loop through vector subindex
// CHECK-NOT: firrtl.circuit "hasloops"
firrtl.circuit "hasloops"   {
  // expected-error @below {{hasloops.{w[3] <- z <- ... <- w[3]}}}
  firrtl.module @hasloops(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
    %w = firrtl.wire  : !firrtl.vector<uint<1>,10>
    %y = firrtl.subindex %w[3]  : !firrtl.vector<uint<1>,10>
    firrtl.connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>
    %0 = firrtl.and %c, %y : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    %z = firrtl.node %0  : !firrtl.uint<1>
    firrtl.connect %y, %z : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %d, %z : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

// Node combinational loop through vector subindex
// CHECK-NOT: firrtl.circuit "hasloops"
firrtl.circuit "hasloops"   {
  // expected-error @below {{hasloops.{bar_a[0] <- b[0] <- bar_b[0] <- bar_a[0]}}}
  firrtl.module @hasloops(out %b: !firrtl.vector<uint<1>, 2>) {
    %bar_a = firrtl.wire : !firrtl.vector<uint<1>, 2>
    %bar_b = firrtl.wire : !firrtl.vector<uint<1>, 2>
    %0 = firrtl.subindex %b[0] : !firrtl.vector<uint<1>, 2>
    %1 = firrtl.subindex %bar_a[0] : !firrtl.vector<uint<1>, 2>
    firrtl.strictconnect %1, %0 : !firrtl.uint<1>
    %4 = firrtl.subindex %bar_b[0] : !firrtl.vector<uint<1>, 2>
    %5 = firrtl.subindex %b[0] : !firrtl.vector<uint<1>, 2>
    firrtl.strictconnect %5, %4 : !firrtl.uint<1>
    %v0 = firrtl.subindex %bar_a[0] : !firrtl.vector<uint<1>, 2>
    %v1 = firrtl.subindex %bar_b[0] : !firrtl.vector<uint<1>, 2>
    firrtl.strictconnect %v1, %v0 : !firrtl.uint<1>
  }
}

// -----

// Combinational loop through instance ports
// CHECK-NOT: firrtl.circuit "hasloops"
firrtl.circuit "hasLoops"  {
  // expected-error @below {{hasLoops.{bar.a[0] <- b[0] <- bar.b[0] <- bar.a[0]}}}
  firrtl.module @hasLoops(out %b: !firrtl.vector<uint<1>, 2>) {
    %bar_a, %bar_b = firrtl.instance bar  @Bar(in a: !firrtl.vector<uint<1>, 2>, out b: !firrtl.vector<uint<1>, 2>)
    %0 = firrtl.subindex %b[0] : !firrtl.vector<uint<1>, 2>
    %1 = firrtl.subindex %bar_a[0] : !firrtl.vector<uint<1>, 2>
    firrtl.strictconnect %1, %0 : !firrtl.uint<1>
    %4 = firrtl.subindex %bar_b[0] : !firrtl.vector<uint<1>, 2>
    %5 = firrtl.subindex %b[0] : !firrtl.vector<uint<1>, 2>
    firrtl.strictconnect %5, %4 : !firrtl.uint<1>
  }
   
  firrtl.module private @Bar(in %a: !firrtl.vector<uint<1>, 2>, out %b: !firrtl.vector<uint<1>, 2>) {
    %0 = firrtl.subindex %a[0] : !firrtl.vector<uint<1>, 2>
    %1 = firrtl.subindex %b[0] : !firrtl.vector<uint<1>, 2>
    firrtl.strictconnect %1, %0 : !firrtl.uint<1>
    %2 = firrtl.subindex %a[1] : !firrtl.vector<uint<1>, 2>
    %3 = firrtl.subindex %b[1] : !firrtl.vector<uint<1>, 2>
    firrtl.strictconnect %3, %2 : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "bundleWire"   {
  // expected-error @below {{bundleWire.{w.foo.bar.baz <- out2 <- x <- w.foo.bar.baz}}}
  firrtl.module @bundleWire(in %arg: !firrtl.bundle<foo: bundle<bar: bundle<baz: uint<1>>, qux: sint<64>>>,
                           out %out1: !firrtl.uint<1>, out %out2: !firrtl.sint<64>) {

    %w = firrtl.wire : !firrtl.bundle<foo: bundle<bar: bundle<baz: sint<64>>, qux: uint<1>>>
    %w0 = firrtl.subfield %w[foo] : !firrtl.bundle<foo: bundle<bar: bundle<baz: sint<64>>, qux: uint<1>>>
    %w0_0 = firrtl.subfield %w0[bar] : !firrtl.bundle<bar: bundle<baz: sint<64>>, qux: uint<1>>
    %w0_0_0 = firrtl.subfield %w0_0[baz] : !firrtl.bundle<baz: sint<64>>
    %x = firrtl.wire  : !firrtl.sint<64>

    %0 = firrtl.subfield %arg[foo] : !firrtl.bundle<foo: bundle<bar: bundle<baz: uint<1>>, qux: sint<64>>>
    %1 = firrtl.subfield %0[bar] : !firrtl.bundle<bar: bundle<baz: uint<1>>, qux: sint<64>>
    %2 = firrtl.subfield %1[baz] : !firrtl.bundle<baz: uint<1>>
    %3 = firrtl.subfield %0[qux] : !firrtl.bundle<bar: bundle<baz: uint<1>>, qux: sint<64>>
    firrtl.connect %w0_0_0, %3 : !firrtl.sint<64>, !firrtl.sint<64>
    firrtl.connect %x, %w0_0_0 : !firrtl.sint<64>, !firrtl.sint<64>
    firrtl.connect %out2, %x : !firrtl.sint<64>, !firrtl.sint<64>
    firrtl.connect %w0_0_0, %out2 : !firrtl.sint<64>, !firrtl.sint<64>
    firrtl.connect %out1, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "registerLoop"   {
  // CHECK: firrtl.module @registerLoop(in %clk: !firrtl.clock)
  firrtl.module @registerLoop(in %clk: !firrtl.clock) {
    %w = firrtl.wire : !firrtl.bundle<a: uint<1>>
    %r = firrtl.reg %clk : !firrtl.clock, !firrtl.bundle<a: uint<1>>
    %0 = firrtl.subfield %w[a]: !firrtl.bundle<a: uint<1>>
    %1 = firrtl.subfield %w[a]: !firrtl.bundle<a: uint<1>>
    %2 = firrtl.subfield %r[a]: !firrtl.bundle<a: uint<1>>
    %3 = firrtl.subfield %r[a]: !firrtl.bundle<a: uint<1>>
    firrtl.connect %2, %0 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %1, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

// Simple combinational loop
// CHECK-NOT: firrtl.circuit "hasloops"
firrtl.circuit "hasloops"   {
  // expected-error @below {{hasloops.{y <- z <- y}}}
  firrtl.module @hasloops(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
    %y = firrtl.wire  : !firrtl.uint<1>
    %z = firrtl.wire  : !firrtl.uint<1>
    firrtl.connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %z, %y : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %y, %z : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %d, %z : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

// Combinational loop through a combinational memory read port
// CHECK-NOT: firrtl.circuit "hasloops"
firrtl.circuit "hasloops"   {
  // expected-error @below {{hasloops.{y <- z <- m.r.data <- m.r.en <- y}}}
  firrtl.module @hasloops(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
    %y = firrtl.wire  : !firrtl.uint<1>
    %z = firrtl.wire  : !firrtl.uint<1>
    firrtl.connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>
    %m_r = firrtl.mem Undefined  {depth = 2 : i64, name = "m", portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
    %0 = firrtl.subfield %m_r[clk] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
    firrtl.connect %0, %clk : !firrtl.clock, !firrtl.clock
    %1 = firrtl.subfield %m_r[addr] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
    %2 = firrtl.subfield %m_r[en] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
    firrtl.connect %2, %y : !firrtl.uint<1>, !firrtl.uint<1>
    %c1_ui = firrtl.constant 1 : !firrtl.uint
    firrtl.connect %2, %c1_ui : !firrtl.uint<1>, !firrtl.uint
    %3 = firrtl.subfield %m_r[data] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
    firrtl.connect %z, %3 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %y, %z : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %d, %z : !firrtl.uint<1>, !firrtl.uint<1>
  }
}


// -----

// Combination loop through an instance
// CHECK-NOT: firrtl.circuit "hasloops"
firrtl.circuit "hasloops"   {
  firrtl.module @thru1(in %in: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
    firrtl.connect %out, %in : !firrtl.uint<1>, !firrtl.uint<1>
  }

  firrtl.module @thru2(in %in: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
    %inner_in, %inner_out = firrtl.instance inner1 @thru1(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    firrtl.connect %inner_in, %in : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %out, %inner_out : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // expected-error @below {{hasloops.{y <- z <- inner2.out <- inner2.in <- y}}}
  firrtl.module @hasloops(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
    %y = firrtl.wire  : !firrtl.uint<1>
    %z = firrtl.wire  : !firrtl.uint<1>
    firrtl.connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>
    %inner_in, %inner_out = firrtl.instance inner2 @thru2(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    firrtl.connect %inner_in, %y : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %z, %inner_out : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %y, %z : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %d, %z : !firrtl.uint<1>, !firrtl.uint<1>
  }
}


// -----

// CHECK: firrtl.circuit "hasloops"
firrtl.circuit "hasloops"  {
  firrtl.module @thru1(in %clk: !firrtl.clock, in %in: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
    %reg = firrtl.reg  %clk  : !firrtl.clock, !firrtl.uint<1>
    firrtl.connect %reg, %in : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %out, %reg : !firrtl.uint<1>, !firrtl.uint<1>
  }
  firrtl.module @thru2(in %clk: !firrtl.clock, in %in: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
    %inner1_clk, %inner1_in, %inner1_out = firrtl.instance inner1  @thru1(in clk: !firrtl.clock, in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    firrtl.connect %inner1_clk, %clk : !firrtl.clock, !firrtl.clock
    firrtl.connect %inner1_in, %in : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %out, %inner1_out : !firrtl.uint<1>, !firrtl.uint<1>
  }
  firrtl.module @hasloops(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
    %y = firrtl.wire   : !firrtl.uint<1>
    %z = firrtl.wire   : !firrtl.uint<1>
    firrtl.connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>
    %inner2_clk, %inner2_in, %inner2_out = firrtl.instance inner2  @thru2(in clk: !firrtl.clock, in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    firrtl.connect %inner2_clk, %clk : !firrtl.clock, !firrtl.clock
    firrtl.connect %inner2_in, %y : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %z, %inner2_out : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %y, %z : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %d, %z : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "subaccess"   {
  // expected-error-re @below {{subaccess.{b[0].wo <- b[{{[0-3]}}].wo}}}
  firrtl.module @subaccess(in %sel1: !firrtl.uint<2>, out %b: !firrtl.vector<bundle<wo: uint<1>, wi: uint<1>>, 4>) {
    %0 = firrtl.subaccess %b[%sel1] : !firrtl.vector<bundle<wo: uint<1>, wi: uint<1>>, 4>, !firrtl.uint<2>
    %1 = firrtl.subfield %0[wo] : !firrtl.bundle<wo: uint<1>, wi: uint<1>>
    %2 = firrtl.subindex %b[0] : !firrtl.vector<bundle<wo: uint<1>, wi: uint<1>>, 4>
    %3 = firrtl.subfield %2[wo]: !firrtl.bundle<wo: uint<1>, wi: uint<1>>
    firrtl.strictconnect %3, %1 : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "subaccess"   {
  // expected-error-re @below {{subaccess.{b[{{[0-3]}}].wo <- b[{{[0-3]}}].wo}}}
  firrtl.module @subaccess(in %sel1: !firrtl.uint<2>, in %sel2: !firrtl.uint<2>, out %b: !firrtl.vector<bundle<wo: uint<1>, wi: uint<1>>, 4>) {
    %0 = firrtl.subaccess %b[%sel1] : !firrtl.vector<bundle<wo: uint<1>, wi: uint<1>>, 4>, !firrtl.uint<2>
    %1 = firrtl.subfield %0[wo] : !firrtl.bundle<wo: uint<1>, wi: uint<1>>
    %2 = firrtl.subaccess %b[%sel2] : !firrtl.vector<bundle<wo: uint<1>, wi: uint<1>>, 4>, !firrtl.uint<2>
    %3 = firrtl.subfield %2[wo]: !firrtl.bundle<wo: uint<1>, wi: uint<1>>
    firrtl.strictconnect %3, %1 : !firrtl.uint<1>
  }
}

// -----

// CHECK: firrtl.circuit "subaccess"   {
firrtl.circuit "subaccess"   {
  firrtl.module @subaccess(in %sel1: !firrtl.uint<2>, out %b: !firrtl.vector<bundle<wo: uint<1>, wi: uint<1>>, 4>) {
    %0 = firrtl.subaccess %b[%sel1] : !firrtl.vector<bundle<wo: uint<1>, wi: uint<1>>, 4>, !firrtl.uint<2>
    %1 = firrtl.subfield %0[wo] : !firrtl.bundle<wo: uint<1>, wi: uint<1>>
    %2 = firrtl.subindex %b[0] : !firrtl.vector<bundle<wo: uint<1>, wi: uint<1>>, 4>
    %3 = firrtl.subfield %2[wi]: !firrtl.bundle<wo: uint<1>, wi: uint<1>>
    firrtl.strictconnect %3, %1 : !firrtl.uint<1>
  }
}

// -----

// CHECK: firrtl.circuit "subaccess"   {
firrtl.circuit "subaccess"   {
  firrtl.module @subaccess(in %sel1: !firrtl.uint<2>, in %sel2: !firrtl.uint<2>, out %b: !firrtl.vector<bundle<wo: uint<1>, wi: uint<1>>, 4>) {
    %0 = firrtl.subaccess %b[%sel1] : !firrtl.vector<bundle<wo: uint<1>, wi: uint<1>>, 4>, !firrtl.uint<2>
    %1 = firrtl.subfield %0[wo] : !firrtl.bundle<wo: uint<1>, wi: uint<1>>
    %2 = firrtl.subaccess %b[%sel2] : !firrtl.vector<bundle<wo: uint<1>, wi: uint<1>>, 4>, !firrtl.uint<2>
    %3 = firrtl.subfield %2[wi]: !firrtl.bundle<wo: uint<1>, wi: uint<1>>
    firrtl.strictconnect %3, %1 : !firrtl.uint<1>
  }
}

// -----

// CHECK: firrtl.circuit "subaccess"   {
firrtl.circuit "subaccess"   {
  firrtl.module @subaccess(in %sel1: !firrtl.uint<2>, in %sel2: !firrtl.uint<2>, out %b: !firrtl.vector<bundle<wo: uint<1>, wi: uint<1>>, 4>) {
    %0 = firrtl.subaccess %b[%sel1] : !firrtl.vector<bundle<wo: uint<1>, wi: uint<1>>, 4>, !firrtl.uint<2>
    %1 = firrtl.subfield %0[wo] : !firrtl.bundle<wo: uint<1>, wi: uint<1>>
    %2 = firrtl.subaccess %b[%sel1] : !firrtl.vector<bundle<wo: uint<1>, wi: uint<1>>, 4>, !firrtl.uint<2>
    %3 = firrtl.subfield %2[wi]: !firrtl.bundle<wo: uint<1>, wi: uint<1>>
    firrtl.strictconnect %3, %1 : !firrtl.uint<1>
  }
}

// -----

// Two input ports share part of the path to an output port.
// CHECK-NOT: firrtl.circuit "revisitOps"
firrtl.circuit "revisitOps"   {
  firrtl.module @thru(in %in1: !firrtl.uint<1>,in %in2: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
    %1 = firrtl.mux(%in1, %in1, %in2)  : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %out, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // expected-error @below {{revisitOps.{inner2.out <- inner2.in2 <- x <- inner2.out}}}
  firrtl.module @revisitOps() {
    %in1, %in2, %out = firrtl.instance inner2 @thru(in in1: !firrtl.uint<1>,in in2: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    %x = firrtl.wire  : !firrtl.uint<1>
    firrtl.connect %in2, %x : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %x, %out : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

// Two input ports and a wire share path to an output port.
// CHECK-NOT: firrtl.circuit "revisitOps"
firrtl.circuit "revisitOps"   {
  firrtl.module @thru(in %in1: !firrtl.vector<uint<1>,2>, in %in2: !firrtl.vector<uint<1>,3>, out %out: !firrtl.vector<uint<1>,2>) {
    %w = firrtl.wire : !firrtl.uint<1>
    %in1_0 = firrtl.subindex %in1[0] : !firrtl.vector<uint<1>,2>
    %in2_1 = firrtl.subindex %in2[1] : !firrtl.vector<uint<1>,3>
    %out_1 = firrtl.subindex %out[1] : !firrtl.vector<uint<1>,2>
    %1 = firrtl.mux(%w, %in1_0, %in2_1)  : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %out_1, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // expected-error @below {{revisitOps.{inner2.out[1] <- inner2.in2[1] <- x <- inner2.out[1]}}}
  firrtl.module @revisitOps() {
    %in1, %in2, %out = firrtl.instance inner2 @thru(in in1: !firrtl.vector<uint<1>,2>, in in2: !firrtl.vector<uint<1>,3>, out out: !firrtl.vector<uint<1>,2>)
    %in1_0 = firrtl.subindex %in1[0] : !firrtl.vector<uint<1>,2>
    %in2_1 = firrtl.subindex %in2[1] : !firrtl.vector<uint<1>,3>
    %out_1 = firrtl.subindex %out[1] : !firrtl.vector<uint<1>,2>
    %x = firrtl.wire  : !firrtl.uint<1>
    firrtl.connect %in2_1, %x : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %x, %out_1 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

// Shared comb path from input ports, ensure that all the paths to the output port are discovered.
// CHECK-NOT: firrtl.circuit "revisitOps"
firrtl.circuit "revisitOps"   {
  firrtl.module @thru(in %in0: !firrtl.vector<uint<1>,2>, in %in1: !firrtl.vector<uint<1>,2>, in %in2: !firrtl.vector<uint<1>,3>, out %out: !firrtl.vector<uint<1>,2>) {
    %w = firrtl.wire : !firrtl.uint<1>
    %in0_0 = firrtl.subindex %in0[0] : !firrtl.vector<uint<1>,2>
    %in1_0 = firrtl.subindex %in1[0] : !firrtl.vector<uint<1>,2>
    %in2_1 = firrtl.subindex %in2[1] : !firrtl.vector<uint<1>,3>
    %out_1 = firrtl.subindex %out[1] : !firrtl.vector<uint<1>,2>
    %1 = firrtl.mux(%w, %in1_0, %in2_1)  : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    %2 = firrtl.mux(%w, %in0_0, %1)  : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %out_1, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // expected-error @below {{revisitOps.{inner2.out[1] <- inner2.in2[1] <- x <- inner2.out[1]}}}
  firrtl.module @revisitOps() {
    %in0, %in1, %in2, %out = firrtl.instance inner2 @thru(in in0: !firrtl.vector<uint<1>,2>, in in1: !firrtl.vector<uint<1>,2>, in in2: !firrtl.vector<uint<1>,3>, out out: !firrtl.vector<uint<1>,2>)
    %in1_0 = firrtl.subindex %in1[0] : !firrtl.vector<uint<1>,2>
    %in2_1 = firrtl.subindex %in2[1] : !firrtl.vector<uint<1>,3>
    %out_1 = firrtl.subindex %out[1] : !firrtl.vector<uint<1>,2>
    %x = firrtl.wire  : !firrtl.uint<1>
    firrtl.connect %in2_1, %x : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %x, %out_1 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

// Comb path from ground type to aggregate.
// CHECK-NOT: firrtl.circuit "scalarToVec"
firrtl.circuit "scalarToVec"   {
  firrtl.module @thru(in %in1: !firrtl.uint<1>, in %in2: !firrtl.vector<uint<1>,3>, out %out: !firrtl.vector<uint<1>,2>) {
    %out_1 = firrtl.subindex %out[1] : !firrtl.vector<uint<1>,2>
    firrtl.connect %out_1, %in1 : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // expected-error @below {{scalarToVec.{inner2.in1 <- x <- inner2.out[1] <- inner2.in1}}}
  firrtl.module @scalarToVec() {
    %in1_0, %in2, %out = firrtl.instance inner2 @thru(in in1: !firrtl.uint<1>, in in2: !firrtl.vector<uint<1>,3>, out out: !firrtl.vector<uint<1>,2>)
    //%in1_0 = firrtl.subindex %in1[0] : !firrtl.vector<uint<1>,2>
    %out_1 = firrtl.subindex %out[1] : !firrtl.vector<uint<1>,2>
    %x = firrtl.wire  : !firrtl.uint<1>
    firrtl.connect %in1_0, %x : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %x, %out_1 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

// Check diagnostic produced if can't name anything on cycle.
// CHECK-NOT: firrtl.circuit "CycleWithoutNames"
firrtl.circuit "CycleWithoutNames"   {
  // expected-error @below {{detected combinational cycle in a FIRRTL module, but unable to find names for any involved values.}}
  firrtl.module @CycleWithoutNames() {
    // expected-note @below {{cycle detected here}}
    %0 = firrtl.wire  : !firrtl.uint<1>
    firrtl.strictconnect %0, %0 : !firrtl.uint<1>
  }
}

// -----

// Check diagnostic if starting point of detected cycle can't be named.
// Try to find something in the cycle we can name and start there.
firrtl.circuit "CycleStartsUnnammed"   {
  // expected-error @below {{sample path: CycleStartsUnnammed.{n <- ... <- n}}}
  firrtl.module @CycleStartsUnnammed() {
    %0 = firrtl.wire  : !firrtl.uint<1>
    %n = firrtl.node %0 : !firrtl.uint<1>
    firrtl.strictconnect %0, %n : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "CycleThroughForceable"   {
  // expected-error @below {{sample path: CycleThroughForceable.{w <- n <- w}}}
  firrtl.module @CycleThroughForceable() {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>
    %n, %n_ref = firrtl.node %w forceable : !firrtl.uint<1>
    firrtl.strictconnect %w, %n : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "Properties"   {
  firrtl.module @Child(in %in: !firrtl.string, out %out: !firrtl.string) {
    firrtl.propassign %out, %in : !firrtl.string
  }
  // expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: Properties.{child0.in <- child0.out <- child0.in}}}
  firrtl.module @Properties() {
    %in, %out = firrtl.instance child0 @Child(in in: !firrtl.string, out out: !firrtl.string)
    firrtl.propassign %in, %out : !firrtl.string
  }
}

// -----
// Incorrect visit of instance op results was resulting in missed cycles.

firrtl.circuit "Bug5442" {
  firrtl.module private @Bar(in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>) {
    firrtl.strictconnect %b, %a : !firrtl.uint<1>
  }
  firrtl.module private @Baz(in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>, out %c_d: !firrtl.uint<1>) {
    firrtl.strictconnect %b, %a : !firrtl.uint<1>
    firrtl.strictconnect %c_d, %a : !firrtl.uint<1>
  }
// expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: Bug5442.{bar.a <- baz.b <- baz.a <- bar.b <- bar.a}}}
  firrtl.module @Bug5442() attributes {convention = #firrtl<convention scalarized>} {
    %bar_a, %bar_b = firrtl.instance bar @Bar(in a: !firrtl.uint<1>, out b: !firrtl.uint<1>)
    %baz_a, %baz_b, %baz_c_d = firrtl.instance baz @Baz(in a: !firrtl.uint<1>, out b: !firrtl.uint<1>, out c_d: !firrtl.uint<1>)
    firrtl.strictconnect %bar_a, %baz_b : !firrtl.uint<1>
    firrtl.strictconnect %baz_a, %bar_b : !firrtl.uint<1>
  }
}
