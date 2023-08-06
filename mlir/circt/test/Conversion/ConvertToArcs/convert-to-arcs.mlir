// RUN: circt-opt %s --convert-to-arcs | FileCheck %s

// CHECK-LABEL: hw.module @Empty
// CHECK-NEXT:    hw.output
// CHECK-NEXT:  }
hw.module @Empty() {
}


// CHECK-LABEL: hw.module @Passthrough(
// CHECK-SAME:    [[TMP:%.+]]: i4) -> (z: i4) {
// CHECK-NEXT:    hw.output [[TMP]]
// CHECK-NEXT:  }
hw.module @Passthrough(%a: i4) -> (z: i4) {
  hw.output %a : i4
}


// CHECK-LABEL: arc.define @CombOnly_arc(
// CHECK-NEXT:    comb.add
// CHECK-NEXT:    comb.xor
// CHECK-NEXT:    comb.xor
// CHECK-NEXT:    comb.mul
// CHECK-NEXT:    arc.output
// CHECK-NEXT:  }

// CHECK-LABEL: hw.module @CombOnly
hw.module @CombOnly(%i0: i4, %i1: i4) -> (z: i4) {
  // CHECK-NEXT: [[TMP:%.+]] = arc.state @CombOnly_arc(%i0, %i1) lat 0
  // CHECK-NEXT: hw.output [[TMP]]
  %0 = comb.add %i0, %i1 : i4
  %1 = comb.xor %0, %i0 : i4
  %2 = comb.xor %0, %i1 : i4
  %3 = comb.mul %1, %2 : i4
  hw.output %3 : i4
}
// CHECK-NEXT: }


// CHECK-LABEL: arc.define @SplitAtConstants_arc(
// CHECK-NEXT:    comb.add
// CHECK-NEXT:    arc.output
// CHECK-NEXT:  }

// CHECK-LABEL: hw.module @SplitAtConstants
hw.module @SplitAtConstants() -> (z: i4) {
  // CHECK-NEXT: %c1_i4 = hw.constant 1
  // CHECK-NEXT: [[TMP:%.+]] = arc.state @SplitAtConstants_arc(%c1_i4) lat 0
  // CHECK-NEXT: hw.output [[TMP]]
  %c1_i4 = hw.constant 1 : i4
  %0 = comb.add %c1_i4, %c1_i4 : i4
  hw.output %0 : i4
}
// CHECK-NEXT: }


// CHECK-LABEL: arc.define @Pipeline_arc(
// CHECK-NEXT:    comb.add
// CHECK-NEXT:    arc.output
// CHECK-NEXT:  }

// CHECK-LABEL: arc.define @Pipeline_arc_0(
// CHECK-NEXT:    comb.xor
// CHECK-NEXT:    arc.output
// CHECK-NEXT:  }

// CHECK-LABEL: arc.define @Pipeline_arc_1(
// CHECK-NEXT:    comb.mul
// CHECK-NEXT:    arc.output
// CHECK-NEXT:  }

// CHECK-LABEL: hw.module @Pipeline
hw.module @Pipeline(%clock: i1, %i0: i4, %i1: i4) -> (z: i4) {
  // CHECK-NEXT: [[S0:%.+]] = arc.state @Pipeline_arc(%i0, %i1) clock %clock lat 1
  // CHECK-NEXT: [[S1:%.+]] = arc.state @Pipeline_arc_0([[S0]], %i0) clock %clock lat 1
  // CHECK-NEXT: [[S2:%.+]] = arc.state @Pipeline_arc_1([[S1]], %i1) lat 0
  // CHECK-NEXT: hw.output [[S2]]
  %0 = comb.add %i0, %i1 : i4
  %1 = seq.compreg %0, %clock : i4
  %2 = comb.xor %1, %i0 : i4
  %3 = seq.compreg %2, %clock : i4
  %4 = comb.mul %3, %i1 : i4
  hw.output %4 : i4
}
// CHECK-NEXT: }


// CHECK-LABEL: arc.define @Reshuffling_arc(%arg0: i4, %arg1: i4)
// CHECK-NEXT:    arc.output %arg0, %arg1
// CHECK-NEXT:  }

// CHECK-LABEL: arc.define @Reshuffling_arc_0(%arg0: i4, %arg1: i4)
// CHECK-NEXT:    arc.output %arg0, %arg1
// CHECK-NEXT:  }

// CHECK-LABEL: hw.module @Reshuffling
hw.module @Reshuffling(%clockA: i1, %clockB: i1) -> (z0: i4, z1: i4, z2: i4, z3: i4) {
  // CHECK-NEXT: hw.instance "x" @Reshuffling2()
  // CHECK-NEXT: arc.state @Reshuffling_arc(%x.z0, %x.z1) clock %clockA lat 1
  // CHECK-NEXT: arc.state @Reshuffling_arc_0(%x.z2, %x.z3) clock %clockB lat 1
  // CHECK-NEXT: hw.output
  %x.z0, %x.z1, %x.z2, %x.z3 = hw.instance "x" @Reshuffling2() -> (z0: i4, z1: i4, z2: i4, z3: i4)
  %4 = seq.compreg %x.z0, %clockA : i4
  %5 = seq.compreg %x.z1, %clockA : i4
  %6 = seq.compreg %x.z2, %clockB : i4
  %7 = seq.compreg %x.z3, %clockB : i4
  hw.output %4, %5, %6, %7 : i4, i4, i4, i4
}
// CHECK-NEXT: }

hw.module.extern private @Reshuffling2() -> (z0: i4, z1: i4, z2: i4, z3: i4)


// CHECK-LABEL: arc.define @FactorOutCommonOps_arc(
// CHECK-NEXT:    comb.xor
// CHECK-NEXT:    arc.output
// CHECK-NEXT:  }

// CHECK-LABEL: arc.define @FactorOutCommonOps_arc_0(
// CHECK-NEXT:    comb.mul
// CHECK-NEXT:    arc.output
// CHECK-NEXT:  }

// CHECK-LABEL: arc.define @FactorOutCommonOps_arc_1(
// CHECK-NEXT:    comb.add
// CHECK-NEXT:    arc.output
// CHECK-NEXT:  }

// CHECK-LABEL: hw.module @FactorOutCommonOps
hw.module @FactorOutCommonOps(%clock: i1, %i0: i4, %i1: i4) -> (o0: i4, o1: i4) {
  // CHECK-DAG: [[T0:%.+]] = arc.state @FactorOutCommonOps_arc_1(%i0, %i1) lat 0
  %0 = comb.add %i0, %i1 : i4
  // CHECK-DAG: [[T1:%.+]] = arc.state @FactorOutCommonOps_arc([[T0]], %i0) clock %clock lat 1
  // CHECK-DAG: [[T2:%.+]] = arc.state @FactorOutCommonOps_arc_0([[T0]], %i1) clock %clock lat 1
  %1 = comb.xor %0, %i0 : i4
  %2 = comb.mul %0, %i1 : i4
  %3 = seq.compreg %1, %clock : i4
  %4 = seq.compreg %2, %clock : i4
  // CHECK-NEXT: hw.output [[T1]], [[T2]]
  hw.output %3, %4 : i4, i4
}
// CHECK-NEXT: }


// CHECK-LABEL: arc.define @SplitAtInstance_arc(
// CHECK-NEXT:    comb.mul
// CHECK-NEXT:    arc.output
// CHECK-NEXT:  }

// CHECK-LABEL: arc.define @SplitAtInstance_arc_0(
// CHECK-NEXT:    comb.shl
// CHECK-NEXT:    arc.output
// CHECK-NEXT:  }

// CHECK-LABEL: hw.module @SplitAtInstance(
hw.module @SplitAtInstance(%a: i4) -> (z: i4) {
  // CHECK-DAG: [[T0:%.+]] = arc.state @SplitAtInstance_arc(%a) lat 0
  // CHECK-DAG: [[T1:%.+]] = hw.instance "x" @SplitAtInstance2(a: [[T0]]: i4)
  // CHECK-DAG: [[T2:%.+]] = arc.state @SplitAtInstance_arc_0([[T1]]) lat 0
  %0 = comb.mul %a, %a : i4
  %1 = hw.instance "x" @SplitAtInstance2(a: %0: i4) -> (z: i4)
  %2 = comb.shl %1, %1 : i4
  // CHECK-NEXT: hw.output [[T2]]
  hw.output %2 : i4
}
// CHECK-NEXT: }

hw.module.extern private @SplitAtInstance2(%a: i4) -> (z: i4)


// CHECK-LABEL: hw.module @AbsorbNames
hw.module @AbsorbNames(%clock: i1) -> () {
  // CHECK-NEXT: %x.z0, %x.z1 = hw.instance "x" @AbsorbNames2()
  // CHECK-NEXT: arc.state @AbsorbNames_arc(%x.z0, %x.z1) clock %clock lat 1
  // CHECK-SAME:   {names = ["myRegA", "myRegB"]}
  // CHECK-NEXT: hw.output
  %x.z0, %x.z1 = hw.instance "x" @AbsorbNames2() -> (z0: i4, z1: i4)
  %myRegA = seq.compreg %x.z0, %clock : i4
  %myRegB = seq.compreg %x.z1, %clock : i4
}
// CHECK-NEXT: }

hw.module.extern @AbsorbNames2() -> (z0: i4, z1: i4)

// CHECK:   arc.define @[[TRIVIAL_ARC:.+]]([[ARG0:%.+]]: i4)
// CHECK-NEXT:     arc.output [[ARG0]]
// CHECK-NEXT:  }

// CHECK-LABEL: hw.module @Trivial(
hw.module @Trivial(%clock: i1, %i0: i4, %reset: i1) -> (out: i4) {
  // CHECK: [[RES0:%.+]] = arc.state @[[TRIVIAL_ARC]](%i0) clock %clock reset %reset lat 1 {names = ["foo"]
  // CHECK-NEXT: hw.output [[RES0:%.+]]
  %0 = hw.constant 0 : i4
  %foo = seq.compreg %i0, %clock, %reset, %0 : i4
  hw.output %foo : i4
}
// CHECK-NEXT: }

// CHECK-NEXT:   arc.define @[[NONTRIVIAL_ARC_0:.+]]([[ARG0_1:%.+]]: i4)
// CHECK-NEXT:     arc.output [[ARG0_1]]
// CHECK-NEXT:  }

// CHECK-NEXT:   arc.define @[[NONTRIVIAL_ARC_1:.+]]([[ARG0_2:%.+]]: i4)
// CHECK-NEXT:     arc.output [[ARG0_2]]
// CHECK-NEXT:  }

// CHECK-LABEL: hw.module @NonTrivial(
hw.module @NonTrivial(%clock: i1, %i0: i4, %reset1: i1, %reset2: i1) -> (out1: i4, out2: i4) {
  // CHECK: [[RES2:%.+]] = arc.state @[[NONTRIVIAL_ARC_0]](%i0) clock %clock reset %reset1 lat 1 {names = ["foo"]
  // CHECK-NEXT: [[RES3:%.+]] = arc.state @[[NONTRIVIAL_ARC_1]](%i0) clock %clock reset %reset2 lat 1 {names = ["bar"]
  // CHECK-NEXT: hw.output [[RES2]], [[RES3]]
  %0 = hw.constant 0 : i4
  %foo = seq.compreg %i0, %clock, %reset1, %0 : i4
  %bar = seq.compreg %i0, %clock, %reset2, %0 : i4
  hw.output %foo, %bar : i4, i4
}
// CHECK-NEXT: }
