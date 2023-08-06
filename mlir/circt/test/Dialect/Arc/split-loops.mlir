// RUN: circt-opt %s --arc-split-loops | FileCheck %s

// CHECK-LABEL: hw.module @Simple(
hw.module @Simple(%a: i4, %b: i4) -> (x: i4, y: i4) {
  // CHECK-NEXT: %0 = arc.state @SimpleArc_split_0(%a, %b)
  // CHECK-NEXT: %1 = arc.state @SimpleArc_split_1(%0, %a)
  // CHECK-NEXT: %2 = arc.state @SimpleArc_split_2(%0, %b)
  // CHECK-NEXT: hw.output %1, %2
  %0:2 = arc.state @SimpleArc(%a, %b) lat 0 : (i4, i4) -> (i4, i4)
  hw.output %0#0, %0#1 : i4, i4
}
// CHECK-NEXT: }

// CHECK-LABEL: arc.define @SimpleArc_split_0(%arg0: i4, %arg1: i4)
// CHECK-NEXT:    %0 = comb.and %arg0, %arg1
// CHECK-NEXT:    arc.output %0
// CHECK-NEXT:  }
// CHECK-LABEL: arc.define @SimpleArc_split_1(%arg0: i4, %arg1: i4)
// CHECK-NEXT:    %0 = comb.add %arg0, %arg1
// CHECK-NEXT:    arc.output %0
// CHECK-NEXT:  }
// CHECK-LABEL: arc.define @SimpleArc_split_2(%arg0: i4, %arg1: i4)
// CHECK-NEXT:    %0 = comb.mul %arg0, %arg1
// CHECK-NEXT:    arc.output %0
// CHECK-NEXT:  }
// CHECK-NOT:   arc.define @SimplerArc(
arc.define @SimpleArc(%arg0: i4, %arg1: i4) -> (i4, i4) {
  %0 = comb.and %arg0, %arg1 : i4
  %1 = comb.add %0, %arg0 : i4
  %2 = comb.mul %0, %arg1 : i4
  arc.output %1, %2 : i4, i4
}

//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @Unchanged(
hw.module @Unchanged(%a: i4) -> (x: i4, y0: i4, y1: i4) {
  // CHECK-NEXT: %0 = arc.state @UnchangedArc1(%a)
  // CHECK-NEXT: %1:2 = arc.state @UnchangedArc2(%a)
  // CHECK-NEXT: hw.output %0, %1#0, %1#1
  %0 = arc.state @UnchangedArc1(%a) lat 0 : (i4) -> i4
  %1:2 = arc.state @UnchangedArc2(%a) lat 0 : (i4) -> (i4, i4)
  hw.output %0, %1#0, %1#1 : i4, i4, i4
}
// CHECK-NEXT: }

// CHECK-LABEL: arc.define @UnchangedArc1(%arg0: i4)
arc.define @UnchangedArc1(%arg0: i4) -> i4 {
  %0 = comb.mul %arg0, %arg0 : i4
  arc.output %0 : i4
}

// CHECK-LABEL: arc.define @UnchangedArc2(%arg0: i4)
arc.define @UnchangedArc2(%arg0: i4) -> (i4, i4) {
  %true = hw.constant true
  %0, %1 = scf.if %true -> (i4, i4) {
    scf.yield %arg0, %arg0 : i4, i4
  } else {
    scf.yield %arg0, %arg0 : i4, i4
  }
  arc.output %0, %1 : i4, i4
}

//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @Passthrough(
hw.module @Passthrough(%a: i4, %b: i4) -> (x0: i4, x1: i4, y0: i4, y1: i4) {
  // CHECK-NEXT: %0 = arc.state @PassthroughArc2(%a)
  // CHECK-NEXT: hw.output %a, %b, %0, %b
  %0:2 = arc.state @PassthroughArc1(%a, %b) lat 0 : (i4, i4) -> (i4, i4)
  %1:2 = arc.state @PassthroughArc2(%a, %b) lat 0 : (i4, i4) -> (i4, i4)
  hw.output %0#0, %0#1, %1#0, %1#1 : i4, i4, i4, i4
}
// CHECK-NEXT: }

// CHECK-NOT: arc.define @PassthroughArc1(
arc.define @PassthroughArc1(%arg0: i4, %arg1: i4) -> (i4, i4) {
  arc.output %arg0, %arg1 : i4, i4
}

// CHECK-LABEL: arc.define @PassthroughArc2(%arg0: i4)
// CHECK-NEXT:    %0 = comb.mul %arg0, %arg0
// CHECK-NEXT:    arc.output %0
// CHECK-NEXT:  }
arc.define @PassthroughArc2(%arg0: i4, %arg1: i4) -> (i4, i4) {
  %0 = comb.mul %arg0, %arg0 : i4
  arc.output %0, %arg1 : i4, i4
}

//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @NestedRegions(
hw.module @NestedRegions(%a: i4, %b: i4, %c: i4) -> (x: i4, y: i4) {
  // CHECK-NEXT: %0:3 = arc.state @NestedRegionsArc_split_0(%a, %b, %c)
  // CHECK-NEXT: %1 = arc.state @NestedRegionsArc_split_1(%0#0, %0#1)
  // CHECK-NEXT: %2 = arc.state @NestedRegionsArc_split_2(%0#2)
  // CHECK-NEXT: hw.output %1, %2
  %0, %1 = arc.state @NestedRegionsArc(%a, %b, %c) lat 0 : (i4, i4, i4) -> (i4, i4)
  hw.output %0, %1 : i4, i4
}
// CHECK-NEXT: }

// CHECK-LABEL: arc.define @NestedRegionsArc_split_0(%arg0: i4, %arg1: i4, %arg2: i4)
// CHECK-NEXT:    %true = hw.constant true
// CHECK-NEXT:    %0:3 = scf.if %true -> (i4, i4, i4) {
// CHECK-NEXT:      scf.yield %arg0, %arg1, %arg2
// CHECK-NEXT:    } else {
// CHECK-NEXT:      scf.yield %arg2, %arg1, %arg0
// CHECK-NEXT:    }
// CHECK-NEXT:    arc.output %0#0, %0#1, %0#2
// CHECK-NEXT:  }
// CHECK-LABEL: arc.define @NestedRegionsArc_split_1(%arg0: i4, %arg1: i4)
// CHECK-NEXT:    %0 = comb.add %arg0, %arg1
// CHECK-NEXT:    arc.output %0
// CHECK-NEXT:  }
// CHECK-LABEL: arc.define @NestedRegionsArc_split_2(%arg0: i4)
// CHECK-NEXT:    %0 = comb.mul %arg0, %arg0
// CHECK-NEXT:    arc.output %0
// CHECK-NEXT:  }
// CHECK-NOT:   arc.define @NestedRegionsArc(
arc.define @NestedRegionsArc(%arg0: i4, %arg1: i4, %arg2: i4) -> (i4, i4) {
  %true = hw.constant true
  %0, %1, %2 = scf.if %true -> (i4, i4, i4) {
    scf.yield %arg0, %arg1, %arg2 : i4, i4, i4
  } else {
    scf.yield %arg2, %arg1, %arg0 : i4, i4, i4
  }
  %3 = comb.add %0, %1 : i4
  %4 = comb.mul %2, %2 : i4
  arc.output %3, %4 : i4, i4
}

//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @BreakFalseLoops(
hw.module @BreakFalseLoops(%a: i4) -> (x: i4, y: i4) {
  // CHECK-NEXT: %0 = arc.state @BreakFalseLoopsArc_split_0(%a)
  // CHECK-NEXT: %1 = arc.state @BreakFalseLoopsArc_split_1(%0)
  // CHECK-NEXT: %2 = arc.state @BreakFalseLoopsArc_split_0(%3)
  // CHECK-NEXT: %3 = arc.state @BreakFalseLoopsArc_split_1(%a)
  // CHECK-NEXT: hw.output %1, %2
  %0, %1 = arc.state @BreakFalseLoopsArc(%a, %0) lat 0 : (i4, i4) -> (i4, i4)
  %2, %3 = arc.state @BreakFalseLoopsArc(%3, %a) lat 0 : (i4, i4) -> (i4, i4)
  hw.output %1, %2 : i4, i4
}
// CHECK-NEXT: }

// CHECK-LABEL: arc.define @BreakFalseLoopsArc_split_0(%arg0: i4)
// CHECK-NEXT:    %0 = comb.add %arg0, %arg0
// CHECK-NEXT:    arc.output %0
// CHECK-NEXT:  }
// CHECK-LABEL: arc.define @BreakFalseLoopsArc_split_1(%arg0: i4)
// CHECK-NEXT:    %0 = comb.mul %arg0, %arg0
// CHECK-NEXT:    arc.output %0
// CHECK-NEXT:  }
// CHECK-NOT:   arc.define @BreakFalseLoopsArc(
arc.define @BreakFalseLoopsArc(%arg0: i4, %arg1: i4) -> (i4, i4) {
  %0 = comb.add %arg0, %arg0 : i4
  %1 = comb.mul %arg1, %arg1 : i4
  arc.output %0, %1 : i4, i4
}

//===----------------------------------------------------------------------===//
// COM: https://github.com/llvm/circt/issues/4862

// CHECK-LABEL: @SplitDependencyModule
hw.module @SplitDependencyModule(%a: i1) -> (x: i1, y: i1) {
  // CHECK-NEXT: %0 = arc.state @SplitDependency_split_1(%a, %a) lat 0 : (i1, i1) -> i1
  // CHECK-NEXT: %1 = arc.state @SplitDependency_split_0(%a, %a, %0) lat 0 : (i1, i1, i1) -> i1
  // CHECK-NEXT: hw.output %0, %1 : i1, i1
  %0, %1 = arc.state @SplitDependency(%a, %a, %a) lat 0 : (i1, i1, i1) -> (i1, i1)
  hw.output %0, %1 : i1, i1
}
// CHECK-NEXT: }

// CHECK-NEXT: arc.define @SplitDependency_split_0(%arg0: i1, %arg1: i1, %arg2: i1) -> i1 {
// CHECK-NEXT:   %0 = comb.xor %arg0, %arg1 : i1
// CHECK-NEXT:   %1 = comb.xor %0, %arg2 : i1
// CHECK-NEXT:   arc.output %1 : i1
// CHECK-NEXT: }
// CHECK-NEXT: arc.define @SplitDependency_split_1(%arg0: i1, %arg1: i1) -> i1 {
// CHECK-NEXT:   %0 = comb.xor %arg0, %arg1 : i1
// CHECK-NEXT:   arc.output %0 : i1
// CHECK-NEXT: }
// CHECK-NOT:  arc.define @SplitDependency(
arc.define @SplitDependency(%arg0: i1, %arg1: i1, %arg2: i1) -> (i1, i1) {
  %0 = comb.xor %arg0, %arg1 : i1
  %1 = comb.xor %arg2, %arg1 : i1
  %2 = comb.xor %0, %1 : i1
  arc.output %1, %2 : i1, i1
}
