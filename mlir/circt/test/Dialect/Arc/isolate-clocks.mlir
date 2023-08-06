// RUN: circt-opt %s --arc-isolate-clocks | FileCheck %s

// CHECK-LABEL: hw.module @basics
hw.module @basics(%clk0: i1, %clk1: i1, %clk2: i1, %c0: i1, %c1: i1, %in: i32) -> (out0: i32, out1: i32) {
  // COM: check the basic things: clocked ops are grouped properly, lat 0 states
  // COM: are not considered clocked, clock domain materialization
  %0 = comb.and %c0, %c1 : i1
  %1 = arc.state @DummyArc(%in) clock %clk0 enable %0 reset %c1 lat 1 : (i32) -> i32
  %mem = arc.memory <2 x i32, i1>
  arc.memory_write_port %mem, @identity(%c0, %2) clock %clk0 lat 1 : <2 x i32, i1>, i1, i32
  %2 = arc.memory_read_port %mem[%c0] : <2 x i32, i1>
  arc.memory_write_port %mem, @identity(%c1, %1) clock %clk0 lat 1 : <2 x i32, i1>, i1, i32
  %3 = arc.state @DummyArc(%4) clock %clk1 enable %0 reset %c1  lat 1 : (i32) -> i32
  %4 = arc.state @DummyArc(%2) lat 0 : (i32) -> i32
  %5 = arc.state @DummyArc(%4) clock %clk2 lat 1 : (i32) -> i32
  hw.output %3, %5 : i32, i32

  // CHECK-NEXT: [[V0:%.+]] = comb.and %c0, %c1 : i1
  // CHECK-NEXT: [[MEM:%.+]] = arc.memory <2 x i32, i1>
  // CHECK-NEXT: [[V6:%.+]] = arc.memory_read_port [[MEM]][%c0] : <2 x i32, i1>
  // CHECK-NEXT: [[V1:%.+]] = arc.state @DummyArc([[V6]]) lat 0 : (i32) -> i32
  // CHECK-NEXT: arc.clock_domain ([[MEM]], %c1, %c0, [[V6]], [[V0]], %in) clock %clk0 : (!arc.memory<2 x i32, i1>, i1, i1, i32, i1, i32) -> () {
  // CHECK-NEXT: ^bb0(%arg0: !arc.memory<2 x i32, i1>, %arg1: i1, %arg2: i1, %arg3: i32, %arg4: i1, %arg5: i32):
  // CHECK-NEXT:   arc.memory_write_port %arg0, @identity(%arg1, [[V7:%.+]]) lat 1 :
  // CHECK-NEXT:   arc.memory_write_port %arg0, @identity(%arg2, %arg3) lat 1 :
  // CHECK-NEXT:   [[V7]] = arc.state @DummyArc(%arg5) enable %arg4 reset %arg1 lat 1 : (i32) -> i32
  // CHECK-NEXT: }
  // CHECK-NEXT: [[V3:%.+]] = arc.clock_domain ([[V0]], %c1, [[V1]]) clock %clk1 : (i1, i1, i32) -> i32 {
  // CHECK-NEXT: ^bb0(%arg0: i1, %arg1: i1, %arg2: i32):
  // CHECK-NEXT:   [[V5:%.+]] = arc.state @DummyArc(%arg2) enable %arg0 reset %arg1 lat 1 : (i32) -> i32
  // CHECK-NEXT:   arc.output [[V5]] : i32
  // CHECK-NEXT: }
  // CHECK-NEXT: [[V4:%.+]] = arc.clock_domain ([[V1]]) clock %clk2 : (i32) -> i32 {
  // CHECK-NEXT: ^bb0(%arg0: i32):
  // CHECK-NEXT:   [[V5:%.+]] = arc.state @DummyArc(%arg0) lat 1 : (i32) -> i32
  // CHECK-NEXT:   arc.output [[V5]] : i32
  // CHECK-NEXT: }
  // CHECK-NEXT: hw.output [[V3]], [[V4]] : i32, i32
}
arc.define @DummyArc(%arg0: i32) -> i32 {
  arc.output %arg0 : i32
}
arc.define @identity(%arg0: i1, %arg1: i32) -> (i1, i32) {
  arc.output %arg0, %arg1 : i1, i32
}

// CHECK-LABEL: hw.module @preexistingClockDomain
hw.module @preexistingClockDomain(%clk0: i1, %clk1: i1, %in: i32) -> (out0: i32, out1: i32) {
  // COM: the pass also works when there are already clock domains present, but
  // COM: it creates new clock domain ops for all clocks of clocked operations
  // COM: outside of the existing domains instead of merging them.
  %0 = arc.state @DummyArc(%in) clock %clk0 lat 1 : (i32) -> i32
  %1 = arc.clock_domain (%0) clock %clk0 : (i32) -> i32 {
  ^bb0(%arg0: i32):
    %2 = arc.state @DummyArc(%arg0) lat 1 : (i32) -> i32
    arc.output %2 : i32
  }
  %2 = arc.state @DummyArc(%in) clock %clk0 lat 1 : (i32) -> i32
  %3 = arc.clock_domain (%2) clock %clk1 : (i32) -> i32 {
  ^bb0(%arg2: i32):
    %4 = arc.state @DummyArc(%arg2) lat 1 : (i32) -> i32
    arc.output %4 : i32
  }
  %4 = arc.state @DummyArc(%1) clock %clk0 lat 1 : (i32) -> i32
  %5 = arc.state @DummyArc(%3) clock %clk0 lat 1 : (i32) -> i32
  %6 = arc.state @DummyArc2(%1, %3) clock %clk0 lat 1 : (i32, i32) -> i32
  hw.output %4, %6 : i32, i32

  // CHECK-NEXT: [[V0:%.+]] = arc.clock_domain ([[V2:%.+]]#3) clock %clk0 : (i32) -> i32 {
  // CHECK-NEXT: ^bb0(%arg0: i32):
  // CHECK-NEXT:   [[V3:%.+]] = arc.state @DummyArc(%arg0) lat 1 : (i32) -> i32
  // CHECK-NEXT:   arc.output [[V3]] : i32
  // CHECK-NEXT: }
  // CHECK-NEXT: [[V1:%.+]] = arc.clock_domain ([[V2]]#2) clock %clk1 : (i32) -> i32 {
  // CHECK-NEXT: ^bb0(%arg0: i32):
  // CHECK-NEXT:   [[V3:%.+]] = arc.state @DummyArc(%arg0) lat 1 : (i32) -> i32
  // CHECK-NEXT:   arc.output [[V3]] : i32
  // CHECK-NEXT: }
  // CHECK-NEXT: [[V2]]:4 = arc.clock_domain ([[V0]], [[V1]], %in) clock %clk0 : (i32, i32, i32) -> (i32, i32, i32, i32) {
  // CHECK-NEXT: ^bb0(%arg0: i32, %arg1: i32, %arg2: i32):
  // CHECK-NEXT:   [[V3:%.+]] = arc.state @DummyArc2(%arg0, %arg1) lat 1 : (i32, i32) -> i32
  // CHECK-NEXT:   [[V4:%.+]] = arc.state @DummyArc(%arg1) lat 1 : (i32) -> i32
  // CHECK-NEXT:   [[V5:%.+]] = arc.state @DummyArc(%arg0) lat 1 : (i32) -> i32
  // CHECK-NEXT:   [[V6:%.+]] = arc.state @DummyArc(%arg2) lat 1 : (i32) -> i32
  // CHECK-NEXT:   [[V7:%.+]] = arc.state @DummyArc(%arg2) lat 1 : (i32) -> i32
  // CHECK-NEXT:   arc.output [[V3]], [[V5]], [[V6]], [[V7]] : i32, i32, i32, i32
  // CHECK-NEXT: }
  // CHECK-NEXT: hw.output [[V2]]#1, [[V2]]#0 : i32, i32
}
arc.define @DummyArc2(%arg0: i32, %arg1: i32) -> i32 {
  arc.output %arg0 : i32
}
