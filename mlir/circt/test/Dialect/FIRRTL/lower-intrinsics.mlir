// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-intrinsics))' %s   | FileCheck %s

// CHECK-LABEL: "Foo"
firrtl.circuit "Foo" {
  // CHECK-NOT: NameDoesNotMatter
  firrtl.extmodule @NameDoesNotMatter(in i : !firrtl.clock, out size : !firrtl.uint<32>) attributes
                                     {annotations = [{class = "circt.Intrinsic", intrinsic = "circt.sizeof"}]}
  // CHECK-NOT: NameDoesNotMatter2
  firrtl.extmodule @NameDoesNotMatter2(in i : !firrtl.clock, out found : !firrtl.uint<1>) attributes
                                     {annotations = [{class = "circt.Intrinsic", intrinsic = "circt.isX"}]}
  // CHECK-NOT: NameDoesNotMatter3
  firrtl.extmodule @NameDoesNotMatter3<FORMAT: none = "foo">(out found : !firrtl.uint<1>) attributes
                                     {annotations = [{class = "circt.Intrinsic", intrinsic = "circt.plusargs.test"}]}
  // CHECK-NOT: NameDoesNotMatter4
  firrtl.extmodule @NameDoesNotMatter4<FORMAT: none = "foo">(out found : !firrtl.uint<1>, out result: !firrtl.uint<5>) attributes
                                     {annotations = [{class = "circt.Intrinsic", intrinsic = "circt.plusargs.value"}]}

  // CHECK: Foo
  firrtl.module @Foo(in %clk : !firrtl.clock, out %s : !firrtl.uint<32>, out %io1 : !firrtl.uint<1>, out %io2 : !firrtl.uint<1>, out %io3 : !firrtl.uint<1>, out %io4 : !firrtl.uint<5>) {
    %i1, %size = firrtl.instance "" @NameDoesNotMatter(in i : !firrtl.clock, out size : !firrtl.uint<32>)
    // CHECK-NOT: NameDoesNotMatter
    // CHECK: firrtl.int.sizeof
    firrtl.strictconnect %i1, %clk : !firrtl.clock
    firrtl.strictconnect %s, %size : !firrtl.uint<32>

    %i2, %found2 = firrtl.instance "" @NameDoesNotMatter2(in i : !firrtl.clock, out found : !firrtl.uint<1>)
    // CHECK-NOT: NameDoesNotMatter2
    // CHECK: firrtl.int.isX
    firrtl.strictconnect %i2, %clk : !firrtl.clock
    firrtl.strictconnect %io1, %found2 : !firrtl.uint<1>

    %found3 = firrtl.instance "" @NameDoesNotMatter3(out found : !firrtl.uint<1>)
    // CHECK-NOT: NameDoesNotMatter3
    // CHECK: firrtl.int.plusargs.test "foo"
    firrtl.strictconnect %io2, %found3 : !firrtl.uint<1>

    %found4, %result1 = firrtl.instance "" @NameDoesNotMatter4(out found : !firrtl.uint<1>, out result: !firrtl.uint<5>)
    // CHECK-NOT: NameDoesNotMatter4
    // CHECK: firrtl.int.plusargs.value "foo" : !firrtl.uint<5>
    firrtl.strictconnect %io3, %found4 : !firrtl.uint<1>
    firrtl.strictconnect %io4, %result1 : !firrtl.uint<5>
  }

  // CHECK-NOT: NameDoesNotMatte5
  firrtl.intmodule @NameDoesNotMatter5(in i : !firrtl.clock, out size : !firrtl.uint<32>) attributes
                                     {intrinsic = "circt.sizeof"}
  // CHECK-NOT: NameDoesNotMatter6
  firrtl.intmodule @NameDoesNotMatter6(in i : !firrtl.clock, out found : !firrtl.uint<1>) attributes
                                     {intrinsic = "circt.isX"}
  // CHECK-NOT: NameDoesNotMatter7
  firrtl.intmodule @NameDoesNotMatter7<FORMAT: none = "foo">(out found : !firrtl.uint<1>) attributes
                                     {intrinsic = "circt.plusargs.test"}
  // CHECK-NOT: NameDoesNotMatter8
  firrtl.intmodule @NameDoesNotMatter8<FORMAT: none = "foo">(out found : !firrtl.uint<1>, out result: !firrtl.uint<5>) attributes
                                     {intrinsic = "circt.plusargs.value"}

  // CHECK: Bar
  firrtl.module @Bar(in %clk : !firrtl.clock, out %s : !firrtl.uint<32>, out %io1 : !firrtl.uint<1>, out %io2 : !firrtl.uint<1>, out %io3 : !firrtl.uint<1>, out %io4 : !firrtl.uint<5>) {
    %i1, %size = firrtl.instance "" @NameDoesNotMatter5(in i : !firrtl.clock, out size : !firrtl.uint<32>)
    // CHECK-NOT: NameDoesNotMatter5
    // CHECK: firrtl.int.sizeof
    firrtl.strictconnect %i1, %clk : !firrtl.clock
    firrtl.strictconnect %s, %size : !firrtl.uint<32>

    %i2, %found2 = firrtl.instance "" @NameDoesNotMatter6(in i : !firrtl.clock, out found : !firrtl.uint<1>)
    // CHECK-NOT: NameDoesNotMatter6
    // CHECK: firrtl.int.isX
    firrtl.strictconnect %i2, %clk : !firrtl.clock
    firrtl.strictconnect %io1, %found2 : !firrtl.uint<1>

    %found3 = firrtl.instance "" @NameDoesNotMatter7(out found : !firrtl.uint<1>)
    // CHECK-NOT: NameDoesNotMatter7
    // CHECK: firrtl.int.plusargs.test "foo"
    firrtl.strictconnect %io2, %found3 : !firrtl.uint<1>

    %found4, %result1 = firrtl.instance "" @NameDoesNotMatter8(out found : !firrtl.uint<1>, out result: !firrtl.uint<5>)
    // CHECK-NOT: NameDoesNotMatter8
    // CHECK: firrtl.int.plusargs.value "foo" : !firrtl.uint<5>
    firrtl.strictconnect %io3, %found4 : !firrtl.uint<1>
    firrtl.strictconnect %io4, %result1 : !firrtl.uint<5>
  }

  // CHECK-NOT: ClockGate0
  // CHECK-NOT: ClockGate1
  firrtl.extmodule @ClockGate0(in in: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock) attributes {annotations = [{class = "circt.Intrinsic", intrinsic = "circt.clock_gate"}]}
  firrtl.intmodule @ClockGate1(in in: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock) attributes {intrinsic = "circt.clock_gate"}

  // CHECK: ClockGate
  firrtl.module @ClockGate(in %clk: !firrtl.clock, in %en: !firrtl.uint<1>) {
    // CHECK-NOT: ClockGate0
    // CHECK: firrtl.int.clock_gate
    %in1, %en1, %out1 = firrtl.instance "" @ClockGate0(in in: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock)
    firrtl.strictconnect %in1, %clk : !firrtl.clock
    firrtl.strictconnect %en1, %en : !firrtl.uint<1>

    // CHECK-NOT: ClockGate1
    // CHECK: firrtl.int.clock_gate
    %in2, %en2, %out2 = firrtl.instance "" @ClockGate1(in in: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock)
    firrtl.strictconnect %in2, %clk : !firrtl.clock
    firrtl.strictconnect %en2, %en : !firrtl.uint<1>
  }

  // CHECK-NOT: LTLAnd
  // CHECK-NOT: LTLOr
  // CHECK-NOT: LTLDelay1
  // CHECK-NOT: LTLDelay2
  // CHECK-NOT: LTLConcat
  // CHECK-NOT: LTLNot
  // CHECK-NOT: LTLImplication
  // CHECK-NOT: LTLEventually
  // CHECK-NOT: LTLClock
  // CHECK-NOT: LTLDisable
  firrtl.intmodule @LTLAnd(in lhs: !firrtl.uint<1>, in rhs: !firrtl.uint<1>, out out: !firrtl.uint<1>) attributes {intrinsic = "circt.ltl.and"}
  firrtl.intmodule @LTLOr(in lhs: !firrtl.uint<1>, in rhs: !firrtl.uint<1>, out out: !firrtl.uint<1>) attributes {intrinsic = "circt.ltl.or"}
  firrtl.intmodule @LTLDelay1<delay: i64 = 42>(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>) attributes {intrinsic = "circt.ltl.delay"}
  firrtl.intmodule @LTLDelay2<delay: i64 = 42, length: i64 = 1337>(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>) attributes {intrinsic = "circt.ltl.delay"}
  firrtl.intmodule @LTLConcat(in lhs: !firrtl.uint<1>, in rhs: !firrtl.uint<1>, out out: !firrtl.uint<1>) attributes {intrinsic = "circt.ltl.concat"}
  firrtl.intmodule @LTLNot(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>) attributes {intrinsic = "circt.ltl.not"}
  firrtl.intmodule @LTLImplication(in lhs: !firrtl.uint<1>, in rhs: !firrtl.uint<1>, out out: !firrtl.uint<1>) attributes {intrinsic = "circt.ltl.implication"}
  firrtl.intmodule @LTLEventually(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>) attributes {intrinsic = "circt.ltl.eventually"}
  firrtl.intmodule @LTLClock(in in: !firrtl.uint<1>, in clock: !firrtl.clock, out out: !firrtl.uint<1>) attributes {intrinsic = "circt.ltl.clock"}
  firrtl.intmodule @LTLDisable(in in: !firrtl.uint<1>, in condition: !firrtl.uint<1>, out out: !firrtl.uint<1>) attributes {intrinsic = "circt.ltl.disable"}

  // CHECK: firrtl.module @LTL()
  firrtl.module @LTL() {
    // CHECK-NOT: LTLAnd
    // CHECK-NOT: LTLOr
    // CHECK: firrtl.int.ltl.and {{%.+}}, {{%.+}} :
    // CHECK: firrtl.int.ltl.or {{%.+}}, {{%.+}} :
    %and.lhs, %and.rhs, %and.out = firrtl.instance "and" @LTLAnd(in lhs: !firrtl.uint<1>, in rhs: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    %or.lhs, %or.rhs, %or.out = firrtl.instance "or" @LTLOr(in lhs: !firrtl.uint<1>, in rhs: !firrtl.uint<1>, out out: !firrtl.uint<1>)

    // CHECK-NOT: LTLDelay1
    // CHECK-NOT: LTLDelay2
    // CHECK: firrtl.int.ltl.delay {{%.+}}, 42 :
    // CHECK: firrtl.int.ltl.delay {{%.+}}, 42, 1337 :
    %delay1.in, %delay1.out = firrtl.instance "delay1" @LTLDelay1(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    %delay2.in, %delay2.out = firrtl.instance "delay2" @LTLDelay2(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)

    // CHECK-NOT: LTLConcat
    // CHECK-NOT: LTLNot
    // CHECK-NOT: LTLImplication
    // CHECK-NOT: LTLEventually
    // CHECK: firrtl.int.ltl.concat {{%.+}}, {{%.+}} :
    // CHECK: firrtl.int.ltl.not {{%.+}} :
    // CHECK: firrtl.int.ltl.implication {{%.+}}, {{%.+}} :
    // CHECK: firrtl.int.ltl.eventually {{%.+}} :
    %concat.lhs, %concat.rhs, %concat.out = firrtl.instance "concat" @LTLConcat(in lhs: !firrtl.uint<1>, in rhs: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    %not.in, %not.out = firrtl.instance "not" @LTLNot(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    %implication.lhs, %implication.rhs, %implication.out = firrtl.instance "implication" @LTLImplication(in lhs: !firrtl.uint<1>, in rhs: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    %eventually.in, %eventually.out = firrtl.instance "eventually" @LTLEventually(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)

    // CHECK-NOT: LTLClock
    // CHECK: firrtl.int.ltl.clock {{%.+}}, {{%.+}} :
    %clock.in, %clock.clock, %clock.out = firrtl.instance "clock" @LTLClock(in in: !firrtl.uint<1>, in clock: !firrtl.clock, out out: !firrtl.uint<1>)

    // CHECK-NOT: LTLDisable
    // CHECK: firrtl.int.ltl.disable {{%.+}}, {{%.+}} :
    %disable.in, %disable.condition, %disable.out = firrtl.instance "disable" @LTLDisable(in in: !firrtl.uint<1>, in condition: !firrtl.uint<1>, out out: !firrtl.uint<1>)
  }

  // CHECK-NOT: VerifAssert1
  // CHECK-NOT: VerifAssert2
  // CHECK-NOT: VerifAssume
  // CHECK-NOT: VerifCover
  firrtl.intmodule @VerifAssert1(in property: !firrtl.uint<1>) attributes {intrinsic = "circt.verif.assert"}
  firrtl.intmodule @VerifAssert2<label: none = "hello">(in property: !firrtl.uint<1>) attributes {intrinsic = "circt.verif.assert"}
  firrtl.intmodule @VerifAssume(in property: !firrtl.uint<1>) attributes {intrinsic = "circt.verif.assume"}
  firrtl.intmodule @VerifCover(in property: !firrtl.uint<1>) attributes {intrinsic = "circt.verif.cover"}

  // CHECK: firrtl.module @Verif()
  firrtl.module @Verif() {
    // CHECK-NOT: VerifAssert1
    // CHECK-NOT: VerifAssert2
    // CHECK-NOT: VerifAssume
    // CHECK-NOT: VerifCover
    // CHECK: firrtl.int.verif.assert {{%.+}} :
    // CHECK: firrtl.int.verif.assert {{%.+}} {label = "hello"} :
    // CHECK: firrtl.int.verif.assume {{%.+}} :
    // CHECK: firrtl.int.verif.cover {{%.+}} :
    %assert1.property = firrtl.instance "assert1" @VerifAssert1(in property: !firrtl.uint<1>)
    %assert2.property = firrtl.instance "assert2" @VerifAssert2(in property: !firrtl.uint<1>)
    %assume.property = firrtl.instance "assume" @VerifAssume(in property: !firrtl.uint<1>)
    %cover.property = firrtl.instance "cover" @VerifCover(in property: !firrtl.uint<1>)
  }

  firrtl.extmodule @Mux2Cell(in sel: !firrtl.uint<1>, in high: !firrtl.uint, in low: !firrtl.uint, out out: !firrtl.uint) attributes {annotations = [{class = "circt.Intrinsic", intrinsic = "circt.mux2cell"}]}
  firrtl.intmodule @Mux4Cell(in sel: !firrtl.uint<2>, in v3: !firrtl.uint, in v2: !firrtl.uint, in v1: !firrtl.uint, in v0: !firrtl.uint, out out: !firrtl.uint) attributes {intrinsic = "circt.mux4cell"}

  // CHECK: firrtl.module @MuxCell()
  firrtl.module @MuxCell() {
    // CHECK: firrtl.int.mux2cell
    // CHECK: firrtl.int.mux4cell
    %sel_0, %high, %low, %out_0 = firrtl.instance "mux2" @Mux2Cell(in sel: !firrtl.uint<1>, in high: !firrtl.uint, in low: !firrtl.uint, out out: !firrtl.uint)
    %sel_1, %v4, %v3, %v2, %v1, %out_1 = firrtl.instance "mux4" @Mux4Cell(in sel: !firrtl.uint<2>, in v3: !firrtl.uint, in v2: !firrtl.uint, in v1: !firrtl.uint, in v0: !firrtl.uint, out out: !firrtl.uint)
  }

  // CHECK-NOT: HBRInt1
  // CHECK-NOT: HBRInt2
  // CHECK-NOT: HBRInt3
  firrtl.intmodule @HBRInt1(in clock: !firrtl.clock, in reset: !firrtl.uint<1>, out out: !firrtl.uint<1>) attributes {intrinsic = "circt.has_been_reset"}
  firrtl.intmodule @HBRInt2(in clock: !firrtl.clock, in reset: !firrtl.asyncreset, out out: !firrtl.uint<1>) attributes {intrinsic = "circt.has_been_reset"}
  firrtl.intmodule @HBRInt3(in clock: !firrtl.clock, in reset: !firrtl.reset, out out: !firrtl.uint<1>) attributes {intrinsic = "circt.has_been_reset"}

  // CHECK: HasBeenReset
  firrtl.module @HasBeenReset(in %clock: !firrtl.clock, in %reset1: !firrtl.uint<1>, in %reset2: !firrtl.asyncreset, in %reset3: !firrtl.reset) {
    // CHECK-NOT: firrtl.instance
    // CHECK: firrtl.int.has_been_reset {{%.+}}, {{%.+}} : !firrtl.uint<1>
    // CHECK-NOT: firrtl.instance
    // CHECK: firrtl.int.has_been_reset {{%.+}}, {{%.+}} : !firrtl.asyncreset
    // CHECK-NOT: firrtl.instance
    // CHECK: firrtl.int.has_been_reset {{%.+}}, {{%.+}} : !firrtl.reset
    %in_clock1, %in_reset1, %hbr1 = firrtl.instance "" @HBRInt1(in clock: !firrtl.clock, in reset: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    %in_clock2, %in_reset2, %hbr2 = firrtl.instance "" @HBRInt2(in clock: !firrtl.clock, in reset: !firrtl.asyncreset, out out: !firrtl.uint<1>)
    %in_clock3, %in_reset3, %hbr3 = firrtl.instance "" @HBRInt3(in clock: !firrtl.clock, in reset: !firrtl.reset, out out: !firrtl.uint<1>)
    firrtl.strictconnect %in_clock1, %clock : !firrtl.clock
    firrtl.strictconnect %in_clock2, %clock : !firrtl.clock
    firrtl.strictconnect %in_clock3, %clock : !firrtl.clock
    firrtl.strictconnect %in_reset1, %reset1 : !firrtl.uint<1>
    firrtl.strictconnect %in_reset2, %reset2 : !firrtl.asyncreset
    firrtl.strictconnect %in_reset3, %reset3 : !firrtl.reset
  }
}
