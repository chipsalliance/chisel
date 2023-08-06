// RUN: circt-opt %s --test-apply-lowering-options="options=emittedLineLength=9001,verifLabels" --export-verilog --verify-diagnostics | FileCheck %s

// CHECK-LABEL: module Labels
hw.module @Labels(%a: i1) {
  // CHECK: foo1: assert property (a);
  // CHECK: foo2: assume property (a);
  // CHECK: foo3: cover property (a);
  verif.assert %a {label = "foo1"} : i1
  verif.assume %a {label = "foo2"} : i1
  verif.cover %a {label = "foo3"} : i1

  // CHECK: bar: assert property (a);
  // CHECK: bar_0: assert property (a);
  verif.assert %a {label = "bar"} : i1
  verif.assert %a {label = "bar"} : i1
}

// CHECK-LABEL: module BasicEmissionNonTemporal
hw.module @BasicEmissionNonTemporal(%a: i1, %b: i1) {
  %0 = comb.and %a, %b : i1
  %1 = comb.or %a, %b : i1
  // CHECK: assert property (a);
  // CHECK: assume property (a & b);
  // CHECK: cover property (a | b);
  verif.assert %a : i1
  verif.assume %0 : i1
  verif.cover %1 : i1

  // CHECK: initial begin
  sv.initial {
    %2 = comb.xor %a, %b : i1
    %3 = comb.and %a, %b : i1
    // CHECK: assert(a);
    // CHECK: assume(a ^ b);
    // CHECK: cover(a & b);
    verif.assert %a : i1
    verif.assume %2 : i1
    verif.cover %3 : i1
  }
}

// CHECK-LABEL: module BasicEmissionTemporal
hw.module @BasicEmissionTemporal(%a: i1) {
  %p = ltl.not %a : i1
  // CHECK: assert property (not a);
  // CHECK: assume property (not a);
  // CHECK: cover property (not a);
  verif.assert %p : !ltl.property
  verif.assume %p : !ltl.property
  verif.cover %p : !ltl.property

  // CHECK: initial begin
  sv.initial {
    // CHECK: assert property (not a);
    // CHECK: assume property (not a);
    // CHECK: cover property (not a);
    verif.assert %p : !ltl.property
    verif.assume %p : !ltl.property
    verif.cover %p : !ltl.property
  }
}

// CHECK-LABEL: module Sequences
hw.module @Sequences(%clk: i1, %a: i1, %b: i1) {
  // CHECK: assert property (##0 a);
  // CHECK: assert property (##4 a);
  // CHECK: assert property (##[5:6] a);
  // CHECK: assert property (##[7:$] a);
  // CHECK: assert property (##[*] a);
  // CHECK: assert property (##[+] a);
  %d0 = ltl.delay %a, 0, 0 : i1
  %d1 = ltl.delay %a, 4, 0 : i1
  %d2 = ltl.delay %a, 5, 1 : i1
  %d3 = ltl.delay %a, 7 : i1
  %d4 = ltl.delay %a, 0 : i1
  %d5 = ltl.delay %a, 1 : i1
  verif.assert %d0 : !ltl.sequence
  verif.assert %d1 : !ltl.sequence
  verif.assert %d2 : !ltl.sequence
  verif.assert %d3 : !ltl.sequence
  verif.assert %d4 : !ltl.sequence
  verif.assert %d5 : !ltl.sequence

  // CHECK: assert property (a ##0 a);
  // CHECK: assert property (a ##4 a);
  // CHECK: assert property (a ##4 a ##[5:6] a);
  // CHECK: assert property (##4 a ##[5:6] a ##[7:$] a);
  %c0 = ltl.concat %a, %a : i1, i1
  %c1 = ltl.concat %a, %d1 : i1, !ltl.sequence
  %c2 = ltl.concat %a, %d1, %d2 : i1, !ltl.sequence, !ltl.sequence
  %c3 = ltl.concat %d1, %d2, %d3 : !ltl.sequence, !ltl.sequence, !ltl.sequence
  verif.assert %c0 : !ltl.sequence
  verif.assert %c1 : !ltl.sequence
  verif.assert %c2 : !ltl.sequence
  verif.assert %c3 : !ltl.sequence

  // CHECK: assert property (a and b);
  // CHECK: assert property (a ##0 a and a ##4 a);
  // CHECK: assert property (a or b);
  // CHECK: assert property (a ##0 a or a ##4 a);
  %g0 = ltl.and %a, %b : i1, i1
  %g1 = ltl.and %c0, %c1 : !ltl.sequence, !ltl.sequence
  %g2 = ltl.or %a, %b : i1, i1
  %g3 = ltl.or %c0, %c1 : !ltl.sequence, !ltl.sequence
  verif.assert %g0 : !ltl.sequence
  verif.assert %g1 : !ltl.sequence
  verif.assert %g2 : !ltl.sequence
  verif.assert %g3 : !ltl.sequence

  // CHECK: assert property (@(posedge clk) a);
  // CHECK: assert property (@(negedge clk) a);
  // CHECK: assert property (@(edge clk) a);
  // CHECK: assert property (@(posedge clk) ##4 a);
  // CHECK: assert property (b ##0 (@(posedge clk) a));
  %k0 = ltl.clock %a, posedge %clk : i1
  %k1 = ltl.clock %a, negedge %clk : i1
  %k2 = ltl.clock %a, edge %clk : i1
  %k3 = ltl.clock %d1, posedge %clk : !ltl.sequence
  %k4 = ltl.concat %b, %k0 : i1, !ltl.sequence
  verif.assert %k0 : !ltl.sequence
  verif.assert %k1 : !ltl.sequence
  verif.assert %k2 : !ltl.sequence
  verif.assert %k3 : !ltl.sequence
  verif.assert %k4 : !ltl.sequence
}

// CHECK-LABEL: module Properties
hw.module @Properties(%clk: i1, %a: i1, %b: i1) {
  %true = hw.constant true

  // CHECK: assert property (not a);
  %n0 = ltl.not %a : i1
  verif.assert %n0 : !ltl.property

  // CHECK: assert property (a |-> b);
  // CHECK: assert property (a ##1 b |-> not a);
  // CHECK: assert property (a ##1 b |=> not a);
  %i0 = ltl.implication %a, %b : i1, i1
  verif.assert %i0 : !ltl.property
  %i1 = ltl.delay %b, 1, 0 : i1
  %i2 = ltl.concat %a, %i1 : i1, !ltl.sequence
  %i3 = ltl.implication %i2, %n0 : !ltl.sequence, !ltl.property
  verif.assert %i3 : !ltl.property
  %i4 = ltl.delay %true, 1, 0 : i1
  %i5 = ltl.concat %a, %i1, %i4 : i1, !ltl.sequence, !ltl.sequence
  %i6 = ltl.implication %i5, %n0 : !ltl.sequence, !ltl.property
  verif.assert %i6 : !ltl.property

  // CHECK: assert property (s_eventually a);
  %e0 = ltl.eventually %a : i1
  verif.assert %e0 : !ltl.property

  // CHECK: assert property (@(posedge clk) a |-> b);
  // CHECK: assert property (@(posedge clk) a ##1 b |-> (@(negedge b) not a));
  // CHECK: assert property (disable iff (b) not a);
  // CHECK: assert property (disable iff (b) @(posedge clk) a |-> b);
  // CHECK: assert property (@(posedge clk) disable iff (b) not a);
  %k0 = ltl.clock %i0, posedge %clk : !ltl.property
  %k1 = ltl.clock %n0, negedge %b : !ltl.property
  %k2 = ltl.implication %i2, %k1 : !ltl.sequence, !ltl.property
  %k3 = ltl.clock %k2, posedge %clk : !ltl.property
  %k4 = ltl.disable %n0 if %b : !ltl.property
  %k5 = ltl.disable %k0 if %b : !ltl.property
  %k6 = ltl.clock %k4, posedge %clk : !ltl.property
  verif.assert %k0 : !ltl.property
  verif.assert %k3 : !ltl.property
  verif.assert %k4 : !ltl.property
  verif.assert %k5 : !ltl.property
  verif.assert %k6 : !ltl.property
}

// CHECK-LABEL: module Precedence
hw.module @Precedence(%a: i1, %b: i1) {
  // CHECK: assert property ((a or b) and b);
  %a0 = ltl.or %a, %b : i1, i1
  %a1 = ltl.and %a0, %b : !ltl.sequence, i1
  verif.assert %a1 : !ltl.sequence

  // CHECK: assert property (##1 (a or b));
  %d0 = ltl.delay %a0, 1, 0 : !ltl.sequence
  verif.assert %d0 : !ltl.sequence

  // CHECK: assert property (not (a or b));
  %n0 = ltl.not %a0 : !ltl.sequence
  verif.assert %n0 : !ltl.property

  // CHECK: assert property (a and (a |-> b));
  %i0 = ltl.implication %a, %b : i1, i1
  %i1 = ltl.and %a, %i0 : i1, !ltl.property
  verif.assert %i1 : !ltl.property

  // CHECK: assert property ((s_eventually a) and b);
  // CHECK: assert property (b and (s_eventually a));
  %e0 = ltl.eventually %a : i1
  %e1 = ltl.and %e0, %b : !ltl.property, i1
  %e2 = ltl.and %b, %e0 : i1, !ltl.property
  verif.assert %e1 : !ltl.property
  verif.assert %e2 : !ltl.property
}

// CHECK-LABEL: module SystemVerilogSpecExamples
hw.module @SystemVerilogSpecExamples(%clk: i1, %a: i1, %b: i1, %c: i1, %d: i1, %e: i1) {
  // Section 16.7 "Sequences"

  // CHECK: assert property (a ##1 b ##0 c ##1 d);
  %a0 = ltl.delay %b, 1, 0 : i1
  %a1 = ltl.delay %d, 1, 0 : i1
  %a2 = ltl.concat %a, %a0 : i1, !ltl.sequence
  %a3 = ltl.concat %c, %a1 : i1, !ltl.sequence
  %a4 = ltl.concat %a2, %a3 : !ltl.sequence, !ltl.sequence
  verif.assert %a4 : !ltl.sequence

  // Section 16.12.20 "Property examples"

  // CHECK: assert property (@(posedge clk) a |-> b ##1 c ##1 d);
  %b0 = ltl.delay %c, 1, 0 : i1
  %b1 = ltl.concat %b, %b0, %a1 : i1, !ltl.sequence, !ltl.sequence
  %b2 = ltl.implication %a, %b1 : i1, !ltl.sequence
  %b3 = ltl.clock %b2, posedge %clk : !ltl.property
  verif.assert %b3 : !ltl.property

  // CHECK: assert property (@(posedge clk) disable iff (e) a |-> not b ##1 c ##1 d);
  %c0 = ltl.not %b1 : !ltl.sequence
  %c1 = ltl.implication %a, %c0 : i1, !ltl.property
  %c2 = ltl.disable %c1 if %e : !ltl.property
  %c3 = ltl.clock %c2, posedge %clk : !ltl.property
  verif.assert %c3 : !ltl.property

  // CHECK: assert property (##1 a |-> b);
  %d0 = ltl.delay %a, 1, 0 : i1
  %d1 = ltl.implication %d0, %b : !ltl.sequence, i1
  verif.assert %d1 : !ltl.property
}

// CHECK-LABEL: module LivenessExample
hw.module @LivenessExample(%clock: i1, %reset: i1, %isLive: i1) {
  %true = hw.constant true

  // CHECK: wire _GEN = ~isLive;
  // CHECK: assert property (@(posedge clock) disable iff (reset) $fell(reset) & _GEN |-> (s_eventually isLive));
  // CHECK: assume property (@(posedge clock) disable iff (reset) $fell(reset) & _GEN |-> (s_eventually isLive));
  %not_isLive = comb.xor %isLive, %true : i1
  %fell_reset = sv.verbatim.expr "$fell({{0}})"(%reset) : (i1) -> i1
  %0 = comb.and %fell_reset, %not_isLive : i1
  %1 = ltl.eventually %isLive : i1
  %2 = ltl.implication %0, %1 : i1, !ltl.property
  %3 = ltl.disable %2 if %reset : !ltl.property
  %liveness_after_reset = ltl.clock %3, posedge %clock : !ltl.property
  verif.assert %liveness_after_reset : !ltl.property
  verif.assume %liveness_after_reset : !ltl.property

  // CHECK: assert property (@(posedge clock) disable iff (reset) isLive ##1 _GEN |-> (s_eventually isLive));
  // CHECK: assume property (@(posedge clock) disable iff (reset) isLive ##1 _GEN |-> (s_eventually isLive));
  %4 = ltl.delay %not_isLive, 1, 0 : i1
  %5 = ltl.concat %isLive, %4 : i1, !ltl.sequence
  %6 = ltl.implication %5, %1 : !ltl.sequence, !ltl.property
  %7 = ltl.disable %6 if %reset : !ltl.property
  %liveness_after_fall = ltl.clock %7, posedge %clock : !ltl.property
  verif.assert %liveness_after_fall : !ltl.property
  verif.assume %liveness_after_fall : !ltl.property
}

// https://github.com/llvm/circt/issues/5763
// CHECK-LABEL: module Issue5763
hw.module @Issue5763(%a: i3) {
  // CHECK: assert property ((&a) & a[0]);
  %c-1_i3 = hw.constant -1 : i3
  %0 = comb.extract %a from 0 : (i3) -> i1
  %1 = comb.icmp bin eq %a, %c-1_i3 : i3
  %2 = comb.and bin %1, %0 : i1
  verif.assert %2 : i1
}
