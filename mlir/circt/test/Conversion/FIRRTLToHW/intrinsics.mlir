// RUN: circt-opt --lower-firrtl-to-hw --verify-diagnostics %s | FileCheck %s

firrtl.circuit "Intrinsics" {
  // CHECK-LABEL: hw.module @Intrinsics
  firrtl.module @Intrinsics(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>) {
    // CHECK-NEXT: %c0_i32 = hw.constant 0 : i32
    // CHECK-NEXT: %false = hw.constant false 
    // CHECK-NEXT: %x_i1 = sv.constantX : i1
    // CHECK-NEXT: [[T0:%.+]] = comb.icmp bin ceq %a, %x_i1
    // CHECK-NEXT: [[T1:%.+]] = comb.icmp bin ceq %clk, %x_i1
    // CHECK-NEXT: %x0 = hw.wire [[T0]]
    // CHECK-NEXT: %x1 = hw.wire [[T1]]
    %0 = firrtl.int.isX %a : !firrtl.uint<1>
    %1 = firrtl.int.isX %clk : !firrtl.clock
    %x0 = firrtl.node interesting_name %0 : !firrtl.uint<1>
    %x1 = firrtl.node interesting_name %1 : !firrtl.uint<1>

    // CHECK-NEXT: [[FOO_STR:%.*]] = sv.constantStr "foo"
    // CHECK-NEXT: [[FOO_DECL:%.*]] = sv.reg : !hw.inout<i1>
    // CHECK-NEXT: sv.initial {
    // CHECK-NEXT:   [[TMP:%.*]] = sv.system "test$plusargs"([[FOO_STR]])
    // CHECK-NEXT:   sv.bpassign [[FOO_DECL]], [[TMP]]
    // CHECK-NEXT: }
    // CHECK-NEXT: [[FOO:%.*]] = sv.read_inout [[FOO_DECL]]
    // CHECK-NEXT: [[BAR_VALUE_DECL:%.*]] = sv.reg : !hw.inout<i5>
    // CHECK-NEXT: [[BAR_FOUND_DECL:%.*]] = sv.reg : !hw.inout<i1>
    // CHECK-NEXT: sv.ifdef "SYNTHESIS" {
    // CHECK-NEXT:   sv.assign [[BAR_FOUND_DECL]], %false
    // CHECK-NEXT: } else {
    // CHECK-NEXT:   sv.initial {
    // CHECK-NEXT:     [[BAR_STR:%.*]] = sv.constantStr "bar"
    // CHECK-NEXT:     [[TMP:%.*]] = sv.system "value$plusargs"([[BAR_STR]], [[BAR_VALUE_DECL]])
    // CHECK-NEXT:     [[TMP2:%.*]] = comb.icmp bin ne [[TMP]], %c0_i32
    // CHECK-NEXT:     sv.bpassign [[BAR_FOUND_DECL]], [[TMP2]]
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    // CHECK-NEXT: [[BAR_FOUND:%.*]] = sv.read_inout [[BAR_FOUND_DECL]]
    // CHECK-NEXT: [[BAR_VALUE:%.*]] = sv.read_inout [[BAR_VALUE_DECL]]
    // CHECK-NEXT: %x2 = hw.wire [[FOO]]
    // CHECK-NEXT: %x3 = hw.wire [[BAR_FOUND]]
    // CHECK-NEXT: %x4 = hw.wire [[BAR_VALUE]]
    %2 = firrtl.int.plusargs.test "foo"
    %3, %4 = firrtl.int.plusargs.value "bar" : !firrtl.uint<5>
    %x2 = firrtl.node interesting_name %2 : !firrtl.uint<1>
    %x3 = firrtl.node interesting_name %3 : !firrtl.uint<1>
    %x4 = firrtl.node interesting_name %4 : !firrtl.uint<5>
  }

  // CHECK-LABEL: hw.module @ClockGate
  firrtl.module @ClockGate(
    in %clk: !firrtl.clock,
    in %enable: !firrtl.uint<1>,
    in %testEnable: !firrtl.uint<1>,
    out %gated_clk0: !firrtl.clock,
    out %gated_clk1: !firrtl.clock
  ) {
    // CHECK-NEXT: [[CLK0:%.+]] = seq.clock_gate %clk, %enable
    // CHECK-NEXT: [[CLK1:%.+]] = seq.clock_gate %clk, %enable, %testEnable
    // CHECK-NEXT: hw.output [[CLK0]], [[CLK1]]
    %0 = firrtl.int.clock_gate %clk, %enable
    %1 = firrtl.int.clock_gate %clk, %enable, %testEnable
    firrtl.strictconnect %gated_clk0, %0 : !firrtl.clock
    firrtl.strictconnect %gated_clk1, %1 : !firrtl.clock
  }

  // CHECK-LABEL: hw.module @LTLAndVerif
  firrtl.module @LTLAndVerif(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>) {
    // CHECK-NEXT: [[D0:%.+]] = ltl.delay %a, 42 : i1
    // CHECK-NEXT: [[D1:%.+]] = ltl.delay %b, 42, 1337 : i1
    %d0 = firrtl.int.ltl.delay %a, 42 : (!firrtl.uint<1>) -> !firrtl.uint<1>
    %d1 = firrtl.int.ltl.delay %b, 42, 1337 : (!firrtl.uint<1>) -> !firrtl.uint<1>

    // CHECK-NEXT: [[L0:%.+]] = ltl.and [[D0]], [[D1]] : !ltl.sequence, !ltl.sequence
    // CHECK-NEXT: [[L1:%.+]] = ltl.or %a, [[L0]] : i1, !ltl.sequence
    %l0 = firrtl.int.ltl.and %d0, %d1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    %l1 = firrtl.int.ltl.or %a, %l0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>

    // CHECK-NEXT: [[C0:%.+]] = ltl.concat [[D0]], [[L1]] : !ltl.sequence, !ltl.sequence
    %c0 = firrtl.int.ltl.concat %d0, %l1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>

    // CHECK-NEXT: [[N0:%.+]] = ltl.not [[C0]] : !ltl.sequence
    %n0 = firrtl.int.ltl.not %c0 : (!firrtl.uint<1>) -> !firrtl.uint<1>

    // CHECK-NEXT: [[I0:%.+]] = ltl.implication [[C0]], [[N0]] : !ltl.sequence, !ltl.property
    %i0 = firrtl.int.ltl.implication %c0, %n0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>

    // CHECK-NEXT: [[E0:%.+]] = ltl.eventually [[I0]] : !ltl.property
    %e0 = firrtl.int.ltl.eventually %i0 : (!firrtl.uint<1>) -> !firrtl.uint<1>

    // CHECK-NEXT: [[K0:%.+]] = ltl.clock [[I0]], posedge %clk : !ltl.property
    %k0 = firrtl.int.ltl.clock %i0, %clk : (!firrtl.uint<1>, !firrtl.clock) -> !firrtl.uint<1>

    // CHECK-NEXT: [[D2:%.+]] = ltl.disable [[K0]] if %b : !ltl.property
    %d2 = firrtl.int.ltl.disable %k0, %b : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>

    // CHECK-NEXT: verif.assert %a : i1
    // CHECK-NEXT: verif.assert %a label "hello" : i1
    // CHECK-NEXT: verif.assume [[C0]] : !ltl.sequence
    // CHECK-NEXT: verif.assume [[C0]] label "hello" : !ltl.sequence
    // CHECK-NEXT: verif.cover [[K0]] : !ltl.property
    // CHECK-NEXT: verif.cover [[K0]] label "hello" : !ltl.property
    firrtl.int.verif.assert %a : !firrtl.uint<1>
    firrtl.int.verif.assert %a {label = "hello"} : !firrtl.uint<1>
    firrtl.int.verif.assume %c0 : !firrtl.uint<1>
    firrtl.int.verif.assume %c0 {label = "hello"} : !firrtl.uint<1>
    firrtl.int.verif.cover %k0 : !firrtl.uint<1>
    firrtl.int.verif.cover %k0 {label = "hello"} : !firrtl.uint<1>
  }

  // CHECK-LABEL: hw.module @LowerIntrinsicStyle
  firrtl.module @LowerIntrinsicStyle(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>) {
    // Wires can make the lowering really weird. Try some strange setup where
    // the ops are totally backwards. This is tricky to lower since a lot of the
    // LTL ops' result type depends on the inputs, and LowerToHW lowers them
    // before their operands have been lowered (and have the correct LTL type).
    // CHECK-NOT: hw.wire
    %c = firrtl.wire : !firrtl.uint<1>
    %d = firrtl.wire : !firrtl.uint<1>
    %e = firrtl.wire : !firrtl.uint<1>
    %f = firrtl.wire : !firrtl.uint<1>
    %g = firrtl.wire : !firrtl.uint<1>

    // CHECK-NEXT: verif.assert [[E:%.+]] : !ltl.sequence
    // CHECK-NEXT: verif.assert [[F:%.+]] : !ltl.property
    // CHECK-NEXT: verif.assert [[G:%.+]] : !ltl.property
    firrtl.int.verif.assert %e : !firrtl.uint<1>
    firrtl.int.verif.assert %f : !firrtl.uint<1>
    firrtl.int.verif.assert %g : !firrtl.uint<1>

    // !ltl.property
    // CHECK-NEXT: [[G]] = ltl.implication [[E]], [[F]] : !ltl.sequence, !ltl.property
    %4 = firrtl.int.ltl.implication %e, %f : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.strictconnect %g, %4 : !firrtl.uint<1>

    // inferred as !ltl.property
    // CHECK-NEXT: [[F]] = ltl.or %b, [[D:%.+]] : i1, !ltl.property
    %3 = firrtl.int.ltl.or %b, %d : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.strictconnect %f, %3 : !firrtl.uint<1>

    // inferred as !ltl.sequence
    // CHECK-NEXT: [[E]] = ltl.and %b, [[C:%.+]] : i1, !ltl.sequence
    %2 = firrtl.int.ltl.and %b, %c : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.strictconnect %e, %2 : !firrtl.uint<1>

    // !ltl.property
    // CHECK-NEXT: [[D]] = ltl.not %b : i1
    %1 = firrtl.int.ltl.not %b : (!firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.strictconnect %d, %1 : !firrtl.uint<1>

    // !ltl.sequence
    // CHECK-NEXT: [[C]] = ltl.delay %a, 42 : i1
    %0 = firrtl.int.ltl.delay %a, 42 : (!firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.strictconnect %c, %0 : !firrtl.uint<1>
  }

  // CHECK-LABEL: hw.module @HasBeenReset
  firrtl.module @HasBeenReset(
    in %clock: !firrtl.clock,
    in %reset1: !firrtl.uint<1>,
    in %reset2: !firrtl.asyncreset,
    out %hbr1: !firrtl.uint<1>,
    out %hbr2: !firrtl.uint<1>
  ) {
    // CHECK-NEXT: [[TMP1:%.+]] = verif.has_been_reset %clock, sync %reset1
    // CHECK-NEXT: [[TMP2:%.+]] = verif.has_been_reset %clock, async %reset2
    // CHECK-NEXT: hw.output [[TMP1]], [[TMP2]]
    %0 = firrtl.int.has_been_reset %clock, %reset1 : !firrtl.uint<1>
    %1 = firrtl.int.has_been_reset %clock, %reset2 : !firrtl.asyncreset
    firrtl.strictconnect %hbr1, %0 : !firrtl.uint<1>
    firrtl.strictconnect %hbr2, %1 : !firrtl.uint<1>
  }
}
