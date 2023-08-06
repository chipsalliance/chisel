// RUN: split-file %s %t

//--- default
// RUN: circt-opt %t/default --arc-inline | FileCheck %t/default

// CHECK-LABEL: func.func @Simple
func.func @Simple(%arg0: i4, %arg1: i1) -> (i4, i4) {
  // CHECK-NEXT: %0 = comb.and %arg0, %arg0
  // CHECK-NEXT: %1 = arc.state @SimpleB(%arg0) clock %arg1 lat 1
  // CHECK-NEXT: return %0, %1
  %0 = arc.state @SimpleA(%arg0) lat 0 : (i4) -> i4
  %1 = arc.state @SimpleB(%arg0) clock %arg1 lat 1 : (i4) -> i4
  return %0, %1 : i4, i4
}
// CHECK-NEXT:  }
// CHECK-NOT: arc.define @SimpleA
arc.define @SimpleA(%arg0: i4) -> i4 {
  %0 = comb.and %arg0, %arg0 : i4
  arc.output %0 : i4
}
// CHECK-LABEL: arc.define @SimpleB
arc.define @SimpleB(%arg0: i4) -> i4 {
  %0 = comb.xor %arg0, %arg0 : i4
  arc.output %0 : i4
}


hw.module @nestedRegionTest(%arg0: i4, %arg1: i4) -> (out0: i4) {
  %0 = arc.state @sub3(%arg0, %arg1) lat 0 : (i4, i4) -> i4
  hw.output %0 : i4
}

arc.define @sub3(%arg0: i4, %arg1: i4) -> i4 {
  %0 = comb.extract %arg0 from 2 : (i4) -> i1
  %1 = scf.if %0 -> (i4) {
    %2 = comb.xor bin %arg0, %arg1 : i4
    scf.yield %2 : i4
  } else {
    %2 = comb.and bin %arg0, %arg1 : i4
    scf.yield %2 : i4
  }
  arc.output %1 : i4
}

// CHECK-LABEL: hw.module @nestedRegionTest
// CHECK-NEXT: [[EXT:%.+]] = comb.extract %arg0 from 2 : (i4) -> i1
// CHECK-NEXT: [[IFRES:%.+]] = scf.if [[EXT]] -> (i4) {
// CHECK-NEXT:   [[XOR:%.+]] = comb.xor bin %arg0, %arg1 : i4
// CHECK-NEXT:   scf.yield [[XOR]] : i4
// CHECK-NEXT: } else {
// CHECK-NEXT:   [[AND:%.+]] = comb.and bin %arg0, %arg1 : i4
// CHECK-NEXT:   scf.yield [[AND]] : i4
// CHECK-NEXT: }
// CHECK-NEXT: hw.output [[IFRES]] : i4

hw.module @opsInNestedRegionsAreAlsoCounted(%arg0: i4, %arg1: i4) -> (out0: i4, out1: i4) {
  %0 = arc.state @sub4(%arg0, %arg1) lat 0 : (i4, i4) -> i4
  %1 = arc.state @sub4(%arg0, %arg1) lat 0 : (i4, i4) -> i4
  hw.output %0, %1 : i4, i4
}

arc.define @sub4(%arg0: i4, %arg1: i4) -> i4 {
  %0 = comb.extract %arg0 from 2 : (i4) -> i1
  %1 = scf.if %0 -> (i4) {
    %2 = comb.xor bin %arg0, %arg1 : i4
    %3 = comb.and bin %2, %arg1 : i4
    scf.yield %3 : i4
  } else {
    %2 = comb.and bin %arg0, %arg1 : i4
    %3 = comb.or bin %arg0, %2 : i4
    scf.yield %3 : i4
  }
  arc.output %1 : i4
}

// CHECK-LABEL: hw.module @opsInNestedRegionsAreAlsoCounted
// CHECK-NEXT:   [[STATERES1:%.+]] = arc.state @sub4(%arg0, %arg1) lat 0 : (i4, i4) -> i4
// CHECK-NEXT:   [[STATERES2:%.+]] = arc.state @sub4(%arg0, %arg1) lat 0 : (i4, i4) -> i4
// CHECK-NEXT:   hw.output [[STATERES1]], [[STATERES2]] : i4, i4

hw.module @nestedBlockArgumentsTest(%arg0: index, %arg1: i4) -> (out0: i4) {
  %0 = arc.state @sub5(%arg0, %arg1) lat 0 : (index, i4) -> i4
  hw.output %0 : i4
}

arc.define @sub5(%arg0: index, %arg1: i4) -> i4 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = scf.for %iv = %c0 to %arg0 step %c1 iter_args (%i = %arg1) -> (i4) {
    %1 = comb.add %i, %arg1 : i4
    scf.yield %1 : i4
  }
  arc.output %0 : i4
}

// CHECK-LABEL: hw.module @nestedBlockArgumentsTest
// CHECK:      [[RES:%.+]] = scf.for {{%.+}} = %c0 to %arg0 step %c1 iter_args([[A0:%.+]] = %arg1) -> (i4) {
// CHECK-NEXT:   [[SUM:%.+]] = comb.add [[A0]], %arg1 : i4
// CHECK-NEXT:   scf.yield [[SUM]] : i4
// CHECK-NEXT: }
// CHECK-NEXT: hw.output [[RES]] : i4

// CHECK-LABEL: hw.module @TopLevel
hw.module @TopLevel(%clk: i1, %arg0: i32, %arg1: i32) -> (out0: i32, out1: i32, out2: i32, out3: i32) {
  %0:2 = arc.state @inlineIntoArc(%arg0, %arg1) clock %clk lat 1 : (i32, i32) -> (i32, i32)
  %1:2 = arc.state @inlineIntoArc2(%arg0, %arg1) clock %clk lat 1 : (i32, i32) -> (i32, i32)
  hw.output %0#0, %0#1, %1#0, %1#1 : i32, i32, i32, i32
}

// CHECK-LABEL: arc.define @inlineIntoArc
arc.define @inlineIntoArc(%arg0: i32, %arg1: i32) -> (i32, i32) {
  // CHECK-NEXT: %0 = comb.add %arg0, %arg1 : i32
  %0 = arc.call @sub6(%arg0, %arg1) : (i32, i32) -> i32
  // CHECK-NEXT: %1 = comb.add %arg0, %arg1 : i32
  %1 = arc.call @sub6(%arg0, %arg1) : (i32, i32) -> i32
  // CHECK-NEXT: arc.output %0, %1 : i32, i32
  arc.output %0, %1 : i32, i32
}

// CHECK-NOT: arc.define @sub6
arc.define @sub6(%arg0: i32, %arg1: i32) -> i32 {
  %0 = comb.add %arg0, %arg1 : i32
  arc.output %0 : i32
}


// CHECK-LABEL: arc.define @inlineIntoArc
arc.define @inlineIntoArc2(%arg0: i32, %arg1: i32) -> (i32, i32) {
  // CHECK-NEXT: %0 = comb.add %arg0, %arg1 : i32
  %0 = arc.call @sub7(%arg0, %arg1) : (i32, i32) -> i32
  // CHECK-NEXT: %1 = comb.add %arg0, %arg1 : i32
  %1 = arc.call @sub7(%arg0, %arg1) : (i32, i32) -> i32
  // CHECK-NEXT: arc.output %0, %1 : i32, i32
  arc.output %0, %1 : i32, i32
}

// CHECK-NOT: arc.define @sub6
arc.define @sub7(%arg0: i32, %arg1: i32) -> i32 {
  %0 = arc.call @sub8(%arg0, %arg1) : (i32, i32) -> i32
  arc.output %0 : i32
}

// CHECK-NOT: arc.define @sub6
arc.define @sub8(%arg0: i32, %arg1: i32) -> i32 {
  %0 = comb.add %arg0, %arg1 : i32
  arc.output %0 : i32
}

// CHECK-NOT: arc.define @ToBeRemoved1
arc.define @ToBeRemoved1(%arg0: i32) -> i32 {
  %0 = arc.call @ToBeRemoved2(%arg0) : (i32) -> i32
  %1 = arc.call @ToBeRemoved3(%0) : (i32) -> i32
  arc.output %1 : i32
}

// CHECK-NOT: arc.define @ToBeRemoved2
arc.define @ToBeRemoved2(%arg0: i32) -> i32 {
  %0 = arc.call @ToBeRemoved3(%arg0) : (i32) -> i32
  arc.output %0 : i32
}

// CHECK-NOT: arc.define @ToBeRemoved3
arc.define @ToBeRemoved3(%arg0: i32) -> i32 {
  arc.output %arg0 : i32
}

//--- onlyIntoArcs
// RUN: circt-opt %t/onlyIntoArcs --arc-inline=into-arcs-only=1 | FileCheck %t/onlyIntoArcs

// CHECK-LABEL: hw.module @onlyIntoArcs
hw.module @onlyIntoArcs(%arg0: i4, %arg1: i4) -> (out0: i4) {
  %0 = arc.state @sub1(%arg0, %arg1) lat 0 : (i4, i4) -> i4
  hw.output %0 : i4
}
// CHECK-LABEL: arc.define @sub1
arc.define @sub1(%arg0: i4, %arg1: i4) -> i4 {
  // CHECK-NEXT: comb.add
  %0 = arc.call @sub2(%arg0, %arg1) : (i4, i4) -> i4
  arc.output %0 : i4
}
// CHECK-NOT: arc.define @sub2
arc.define @sub2(%arg0: i4, %arg1: i4) -> i4 {
  %0 = comb.add %arg0, %arg1 : i4
  arc.output %0 : i4
}
