// RUN: circt-opt %s --arc-canonicalizer | FileCheck %s

//===----------------------------------------------------------------------===//
// Remove Passthrough calls
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @passthoughChecks
hw.module @passthoughChecks(%in0: i1, %in1: i1) -> (out0: i1, out1: i1, out2: i1, out3: i1, out4: i1, out5: i1, out6: i1, out7: i1, out8: i1, out9: i1) {
  %0:2 = arc.call @passthrough(%in0, %in1) : (i1, i1) -> (i1, i1)
  %1:2 = arc.call @noPassthrough(%in0, %in1) : (i1, i1) -> (i1, i1)
  %2:2 = arc.state @passthrough(%in0, %in1) lat 0 : (i1, i1) -> (i1, i1)
  %3:2 = arc.state @noPassthrough(%in0, %in1) lat 0 : (i1, i1) -> (i1, i1)
  %4:2 = arc.state @passthrough(%in0, %in1) clock %in0 lat 1 : (i1, i1) -> (i1, i1)
  hw.output %0#0, %0#1, %1#0, %1#1, %2#0, %2#1, %3#0, %3#1, %4#0, %4#1 : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
  // CHECK-NEXT: [[V0:%.+]]:2 = arc.call @noPassthrough(%in0, %in1) :
  // CHECK-NEXT: [[V1:%.+]]:2 = arc.state @noPassthrough(%in0, %in1) lat 0 :
  // CHECK-NEXT: [[V2:%.+]]:2 = arc.state @passthrough(%in0, %in1) clock %in0 lat 1 :
  // CHECK-NEXT: hw.output %in0, %in1, [[V0]]#0, [[V0]]#1, %in0, %in1, [[V1]]#0, [[V1]]#1, [[V2]]#0, [[V2]]#1 :
}
arc.define @passthrough(%arg0: i1, %arg1: i1) -> (i1, i1) {
  arc.output %arg0, %arg1 : i1, i1
}
arc.define @noPassthrough(%arg0: i1, %arg1: i1) -> (i1, i1) {
  arc.output %arg1, %arg0 : i1, i1
}

//===----------------------------------------------------------------------===//
// MemoryWritePortOp canonicalizer
//===----------------------------------------------------------------------===//

arc.define @memArcFalse(%arg0: i1, %arg1: i32) -> (i1, i32, i1) {
  %false = hw.constant false
  arc.output %arg0, %arg1, %false : i1, i32, i1
}
arc.define @memArcTrue(%arg0: i1, %arg1: i32) -> (i1, i32, i1) {
  %true = hw.constant true
  arc.output %arg0, %arg1, %true : i1, i32, i1
}

// CHECK-LABEL: hw.module @memoryWritePortCanonicalizations
hw.module @memoryWritePortCanonicalizations(%clk: i1, %addr: i1, %data: i32) {
  // CHECK-NEXT: [[MEM:%.+]] = arc.memory <2 x i32, i1>
  %mem = arc.memory <2 x i32, i1>
  arc.memory_write_port %mem, @memArcFalse(%addr, %data) clock %clk enable lat 1 : <2 x i32, i1>, i1, i32
  // CHECK-NEXT: arc.memory_write_port [[MEM]], @memArcTrue_0(%addr, %data) clock %clk lat 1 :
  arc.memory_write_port %mem, @memArcTrue(%addr, %data) clock %clk enable lat 1 : <2 x i32, i1>, i1, i32
  // CHECK-NEXT: arc.memory_write_port [[MEM]], @memArcTrue_0(%addr, %data) clock %clk lat 1 :
  arc.memory_write_port %mem, @memArcTrue(%addr, %data) clock %clk enable lat 1 : <2 x i32, i1>, i1, i32
  %0:3 = arc.state @memArcTrue(%addr, %data) lat 0 : (i1, i32) -> (i1, i32, i1)
  // CHECK-NEXT: hw.output
  hw.output
}

//===----------------------------------------------------------------------===//
// RemoveUnusedArcs
//===----------------------------------------------------------------------===//

// CHECK-NOT: arc.define @unusedArcIsDeleted
arc.define @unusedArcIsDeleted(%arg0: i32, %arg1: i32) -> i32 {
  %0 = arc.call @nestedUnused(%arg0, %arg1) : (i32, i32) -> i32
  arc.output %0 : i32
}
// CHECK-NOT: arc.define @nestedUnused
arc.define @nestedUnused(%arg0: i32, %arg1: i32) -> i32 {
  %0 = comb.add %arg0, %arg1 : i32
  arc.output %0 : i32
}

//===----------------------------------------------------------------------===//
// ICMPCanonicalizer
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @icmpEqCanonicalizer
hw.module @icmpEqCanonicalizer(%arg0: i1, %arg1: i1, %arg2: i1, %arg3: i1, %arg4: i4, %arg5: i4, %arg6: i4, %arg7: i4) -> (out0: i1, out1: i1, out2: i1, out3: i1) {
  // CHECK: [[V0:%.+]] = comb.and bin %arg0, %arg1, %arg2, %arg3 : i1
  %c-1_i4 = hw.constant -1 : i4
  %0 = comb.concat %arg0, %arg1, %arg2, %arg3 : i1, i1, i1, i1
  %1 = comb.icmp bin eq %0, %c-1_i4 : i4

  // CHECK-NEXT: [[V1:%.+]] = comb.or bin %arg0, %arg1, %arg2, %arg3 : i1
  // CHECK-NEXT: [[V2:%.+]] = comb.xor bin [[V1]], %true : i1
  %c0_i4 = hw.constant 0 : i4
  %2 = comb.concat %arg0, %arg1, %arg2, %arg3 : i1, i1, i1, i1
  %3 = comb.icmp bin eq %2, %c0_i4 : i4

  // CHECK-NEXT: [[V3:%.+]] = comb.and bin %arg4, %arg5, %arg6, %arg7 : i4
  // CHECK-NEXT: [[V4:%.+]] = comb.icmp bin eq [[V3]], %c-1_i4 : i4
  %c-1_i16 = hw.constant -1 : i16
  %4 = comb.concat %arg4, %arg5, %arg6, %arg7 : i4, i4, i4, i4
  %5 = comb.icmp bin eq %4, %c-1_i16 : i16

  // CHECK-NEXT: [[V5:%.+]] = comb.or bin %arg4, %arg5, %arg6, %arg7 : i4
  // CHECK-NEXT: [[V6:%.+]] = comb.icmp bin eq [[V5]], %c0_i4 : i4
  %c0_i16 = hw.constant 0 : i16
  %6 = comb.concat %arg4, %arg5, %arg6, %arg7 : i4, i4, i4, i4
  %7 = comb.icmp bin eq %6, %c0_i16 : i16

  // CHECK-NEXT: hw.output [[V0]], [[V2]], [[V4]], [[V6]] :
  hw.output %1, %3, %5, %7 : i1, i1, i1, i1
}

// CHECK-LABEL: hw.module @icmpNeCanonicalizer
hw.module @icmpNeCanonicalizer(%arg0: i1, %arg1: i1, %arg2: i1, %arg3: i1, %arg4: i4, %arg5: i4, %arg6: i4, %arg7: i4) -> (out0: i1, out1: i1, out2: i1, out3: i1) {
  // CHECK: [[V0:%.+]] = comb.or bin %arg0, %arg1, %arg2, %arg3 : i1
  %c0_i4 = hw.constant 0 : i4
  %0 = comb.concat %arg0, %arg1, %arg2, %arg3 : i1, i1, i1, i1
  %1 = comb.icmp bin ne %0, %c0_i4 : i4

  // CHECK-NEXT: [[V1:%.+]] = comb.and bin %arg0, %arg1, %arg2, %arg3 : i1
  // CHECK-NEXT: [[V2:%.+]] = comb.xor bin [[V1]], %true : i1
  %c-1_i4 = hw.constant -1 : i4
  %2 = comb.concat %arg0, %arg1, %arg2, %arg3 : i1, i1, i1, i1
  %3 = comb.icmp bin ne %2, %c-1_i4 : i4

  // CHECK-NEXT: [[V3:%.+]] = comb.or bin %arg4, %arg5, %arg6, %arg7 : i4
  // CHECK-NEXT: [[V4:%.+]] = comb.icmp bin ne [[V3]], %c0_i4 : i4
  %c0_i16 = hw.constant 0 : i16
  %4 = comb.concat %arg4, %arg5, %arg6, %arg7 : i4, i4, i4, i4
  %5 = comb.icmp bin ne %4, %c0_i16 : i16

  // CHECK-NEXT: [[V5:%.+]] = comb.and bin %arg4, %arg5, %arg6, %arg7 : i4
  // CHECK-NEXT: [[V6:%.+]] = comb.icmp bin ne [[V5]], %c-1_i4 : i4
  %c-1_i16 = hw.constant -1 : i16
  %6 = comb.concat %arg4, %arg5, %arg6, %arg7 : i4, i4, i4, i4
  %7 = comb.icmp bin ne %6, %c-1_i16 : i16

  // CHECK-NEXT: hw.output [[V0]], [[V2]], [[V4]], [[V6]] :
  hw.output %1, %3, %5, %7 : i1, i1, i1, i1
}

//===----------------------------------------------------------------------===//
// RemoveUnusedArcArguments
//===----------------------------------------------------------------------===//

// COM: this has to be before @OneOfThreeUsed to check that arguments that
// COM: become unused during the process are removed as well.
// CHECK: arc.define @NestedCall(%arg0: i1) -> i1 {
arc.define @NestedCall(%arg0: i1, %arg1: i1, %arg2: i1) -> i1 {
  // CHECK: arc.call @OneOfThreeUsed(%arg0) : (i1) -> i1
  %0 = arc.call @OneOfThreeUsed(%arg0, %arg1, %arg2) : (i1, i1, i1) -> i1
  arc.output %0 : i1
}

// CHECK-LABEL: arc.define @OneOfThreeUsed(%arg0: i1)
arc.define @OneOfThreeUsed(%arg0: i1, %arg1: i1, %arg2: i1) -> i1 {
  %true = hw.constant true
  %0 = comb.xor %arg1, %true : i1
  // CHECK: arc.output {{%[0-9]+}} :
  arc.output %0 : i1
}

// CHECK: @test1
hw.module @test1 (%arg0: i1, %arg1: i1, %arg2: i1, %clock: i1) -> (out0: i1, out1: i1) {
  // CHECK-NEXT: arc.state @OneOfThreeUsed(%arg1) clock %clock lat 1 : (i1) -> i1
  %0 = arc.state @OneOfThreeUsed(%arg0, %arg1, %arg2) clock %clock lat 1 : (i1, i1, i1) -> i1
  // CHECK-NEXT: arc.state @NestedCall(%arg1)
  %1 = arc.state @NestedCall(%arg0, %arg1, %arg2) clock %clock lat 1 : (i1, i1, i1) -> i1
  hw.output %0, %1 : i1, i1
}

// CHECK-LABEL: arc.define @NoArgsToRemove()
arc.define @NoArgsToRemove() -> i1 {
  %0 = hw.constant 0 : i1
  arc.output %0 : i1
}

// CHECK: @test2
hw.module @test2 () -> (out: i1) {
  // CHECK-NEXT: arc.state @NoArgsToRemove() lat 0 : () -> i1
  %0 = arc.state @NoArgsToRemove() lat 0 : () -> i1
  hw.output %0 : i1
}

//===----------------------------------------------------------------------===//
// SinkArcInputs
//===----------------------------------------------------------------------===//

// CHECK-LABEL: arc.define @SinkSameConstantsArc(%arg0: i4)
arc.define @SinkSameConstantsArc(%arg0: i4, %arg1: i4) -> i4 {
  // CHECK-NEXT: %c2_i4 = hw.constant 2
  // CHECK-NEXT: [[TMP:%.+]] = comb.add %arg0, %c2_i4
  // CHECK-NEXT: arc.output [[TMP]]
  %0 = comb.add %arg0, %arg1 : i4
  arc.output %0 : i4
}
// CHECK-NEXT: }

// CHECK: arc.define @Foo
arc.define @Foo(%arg0: i4) -> i4 {
  // CHECK-NOT: hw.constant
  %k1 = hw.constant 2 : i4
  // CHECK: {{%.+}} = arc.call @SinkSameConstantsArc(%arg0)
  %0 = arc.call @SinkSameConstantsArc(%arg0, %k1) : (i4, i4) -> i4
  arc.output %0 : i4
}

// CHECK: hw.module @SinkSameConstants
hw.module @SinkSameConstants(%x: i4) -> (out0: i4, out1: i4, out2: i4) {
  // CHECK-NOT: hw.constant
  // CHECK-NEXT: %0 = arc.state @SinkSameConstantsArc(%x)
  // CHECK-NEXT: %1 = arc.state @SinkSameConstantsArc(%x)
  // CHECK-NEXT: arc.call
  // CHECK-NEXT: hw.output
  %k1 = hw.constant 2 : i4
  %k2 = hw.constant 2 : i4
  %0 = arc.state @SinkSameConstantsArc(%x, %k1) lat 0 : (i4, i4) -> i4
  %1 = arc.state @SinkSameConstantsArc(%x, %k2) lat 0 : (i4, i4) -> i4
  %2 = arc.call @Foo(%x) : (i4) -> i4
  hw.output %0, %1, %2 : i4, i4, i4
}
// CHECK-NEXT: }


// CHECK-LABEL: arc.define @DontSinkDifferentConstantsArc(%arg0: i4, %arg1: i4)
arc.define @DontSinkDifferentConstantsArc(%arg0: i4, %arg1: i4) -> i4 {
  // CHECK-NEXT: comb.add %arg0, %arg1
  // CHECK-NEXT: arc.output
  %0 = comb.add %arg0, %arg1 : i4
  arc.output %0 : i4
}
// CHECK-NEXT: }

// CHECK-LABEL: hw.module @DontSinkDifferentConstants
hw.module @DontSinkDifferentConstants(%x: i4) -> (out0: i4, out1: i4) {
  // CHECK-NEXT: %c2_i4 = hw.constant 2 : i4
  // CHECK-NEXT: %c3_i4 = hw.constant 3 : i4
  // CHECK-NEXT: %0 = arc.state @DontSinkDifferentConstantsArc(%x, %c2_i4)
  // CHECK-NEXT: %1 = arc.state @DontSinkDifferentConstantsArc(%x, %c3_i4)
  // CHECK-NEXT: hw.output
  %c2_i4 = hw.constant 2 : i4
  %c3_i4 = hw.constant 3 : i4
  %0 = arc.state @DontSinkDifferentConstantsArc(%x, %c2_i4) lat 0 : (i4, i4) -> i4
  %1 = arc.state @DontSinkDifferentConstantsArc(%x, %c3_i4) lat 0 : (i4, i4) -> i4
  hw.output %0, %1 : i4, i4
}
// CHECK-NEXT: }


// CHECK-LABEL: arc.define @DontSinkDifferentConstantsArc1(%arg0: i4, %arg1: i4)
arc.define @DontSinkDifferentConstantsArc1(%arg0: i4, %arg1: i4) -> i4 {
  // CHECK-NEXT: [[TMP:%.+]] = comb.add %arg0, %arg1
  // CHECK-NEXT: arc.output [[TMP]]
  %0 = comb.add %arg0, %arg1 : i4
  arc.output %0 : i4
}
// CHECK-NEXT: }

// CHECK: arc.define @Bar
arc.define @Bar(%arg0: i4) -> i4 {
  // CHECK: %c1_i4 = hw.constant 1
  %k1 = hw.constant 1 : i4
  // CHECK: {{%.+}} = arc.call @DontSinkDifferentConstantsArc1(%arg0, %c1_i4)
  %0 = arc.call @DontSinkDifferentConstantsArc1(%arg0, %k1) : (i4, i4) -> i4
  arc.output %0 : i4
}

// CHECK: hw.module @DontSinkDifferentConstants1
hw.module @DontSinkDifferentConstants1(%x: i4) -> (out0: i4, out1: i4, out2: i4) {
  // CHECK-NEXT: %c2_i4 = hw.constant 2 : i4
  // CHECK-NEXT: %0 = arc.state @DontSinkDifferentConstantsArc1(%x, %c2_i4)
  // CHECK-NEXT: %1 = arc.state @DontSinkDifferentConstantsArc1(%x, %c2_i4)
  // CHECK-NEXT: arc.call
  // CHECK-NEXT: hw.output
  %k1 = hw.constant 2 : i4
  %k2 = hw.constant 2 : i4
  %0 = arc.state @DontSinkDifferentConstantsArc1(%x, %k1) lat 0 : (i4, i4) -> i4
  %1 = arc.state @DontSinkDifferentConstantsArc1(%x, %k2) lat 0 : (i4, i4) -> i4
  %2 = arc.call @Bar(%x) : (i4) -> i4
  hw.output %0, %1, %2 : i4, i4, i4
}
// CHECK-NEXT: }
