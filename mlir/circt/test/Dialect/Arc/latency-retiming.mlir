// RUN: circt-opt %s --arc-latency-retiming | FileCheck %s

// CHECK-LABEL: hw.module @Foo
hw.module @Foo(%clk: i1, %clk2: i1, %en: i1, %rst: i1, %arg0: i32, %arg1: i32) -> (out0: i32, out1: i32, out2: i32, out3: i32, out4: i32, out5: i32, out6: i32, out7: i32, out8: i32, out9: i32, out10: i32, out11: i32) {
  // COM: simple shift-register of depth 3 is merged to one arc.state
  %0 = arc.state @Bar(%arg0) clock %clk lat 1 : (i32) -> i32
  %1 = arc.state @Bar(%0) clock %clk lat 1 : (i32) -> i32
  %2 = arc.state @Bar(%1) clock %clk lat 1 : (i32) -> i32
  // CHECK-NEXT: [[V0:%.+]] = arc.state @Bar(%arg0) clock %clk lat 3 :

  // COM: looks like a shift register from the outside (and actually is), but
  // COM: the arcs are not exactly passthroughs, so don't erase them
  // COM: TODO: a canonicalization pattern could reorder the outputs such that
  // COM:       this also folds to just one state, but inlining will get rid of
  // COM:       them anyways.
  // CHECK-NEXT: [[V1:%.+]]:2 = arc.state @Baz(%arg0, %arg1) lat 0 :
  %3:2 = arc.state @Baz(%arg0, %arg1) clock %clk lat 1 : (i32, i32) -> (i32, i32)
  // CHECK-NEXT: [[V2:%.+]]:2 = arc.state @Baz([[V1]]#0, [[V1]]#1) lat 0 :
  %4:2 = arc.state @Baz(%3#0, %3#1) clock %clk lat 2 : (i32, i32) -> (i32, i32)
  // CHECK-NEXT: [[V3:%.+]]:2 = arc.state @Baz([[V2]]#0, [[V2]]#1) clock %clk lat 6 :
  %5:2 = arc.state @Baz(%4#0, %4#1) clock %clk lat 3 : (i32, i32) -> (i32, i32)

  // COM: a fan-in tree, op that gets the latencies starts with lat 0
  %6 = arc.state @Bar(%arg0) clock %clk lat 1 : (i32) -> i32
  %7 = arc.state @Bar(%arg1) clock %clk lat 1 : (i32) -> i32
  // CHECK-NEXT: [[V4:%.+]]:2 = arc.state @Baz(%arg0, %arg1) lat 0 :
  %8:2 = arc.state @Baz(%6, %7) clock %clk lat 1 : (i32, i32) -> (i32, i32)
  %9 = arc.state @Bar(%arg1) clock %clk lat 2 : (i32) -> i32
  // CHECK-NEXT: [[V5:%.+]]:2 = arc.state @Baz([[V4]]#0, %arg1) clock %clk lat 2 :
  %10:2 = arc.state @Baz(%8#0, %9) lat 0 : (i32, i32) -> (i32, i32)

  // COM: fan-out tree
  // CHECK-NEXT: [[V6:%.+]] = arc.state @Bar(%arg0) clock %clk lat 1 :
  %11 = arc.state @Bar(%arg0) clock %clk lat 1 : (i32) -> i32
  // CHECK-NEXT: arc.state @Bar([[V6]]) clock %clk lat 1 :
  %12 = arc.state @Bar(%11) clock %clk lat 1 : (i32) -> i32
  // CHECK-NEXT: arc.state @Bar([[V6]]) clock %clk lat 1 :
  %13 = arc.state @Bar(%11) clock %clk lat 1 : (i32) -> i32

  // COM: states with names attached are not touched
  %14 = arc.state @Bar(%arg0) clock %clk lat 1 : (i32) -> i32
  // CHECK-NEXT: [[V7:%.+]] = arc.state @Bar(%arg0) clock %clk lat 2 {name = "reg"} :
  %15 = arc.state @Bar(%14) clock %clk lat 1 {name = "reg"} : (i32) -> i32
  // CHECK-NEXT: arc.state @Bar([[V7]]) clock %clk lat 1 :
  %16 = arc.state @Bar(%15) clock %clk lat 1 : (i32) -> i32

  // COM: states with names attached are not touched
  %17 = arc.state @Bar(%arg0) clock %clk lat 1 : (i32) -> i32
  // CHECK-NEXT: [[V8:%.+]] = arc.state @Bar(%arg0) clock %clk lat 2 {names = ["reg"]} :
  %18 = arc.state @Bar(%17) clock %clk lat 1 {names = ["reg"]} : (i32) -> i32
  // CHECK-NEXT: arc.state @Bar([[V8]]) clock %clk lat 1 :
  %19 = arc.state @Bar(%18) clock %clk lat 1 : (i32) -> i32

  // COM: states with enables are not touched
  // CHECK-NEXT: [[V9:%.+]] = arc.state @Bar(%arg0) clock %clk lat 1 :
  %20 = arc.state @Bar(%arg0) clock %clk lat 1 : (i32) -> i32
  // CHECK-NEXT: [[V10:%.+]] = arc.state @Bar([[V9]]) clock %clk enable %en lat 1 :
  %21 = arc.state @Bar(%20) clock %clk enable %en lat 1 : (i32) -> i32
  // CHECK-NEXT: arc.state @Bar([[V10]]) clock %clk lat 1 :
  %22 = arc.state @Bar(%21) clock %clk lat 1 : (i32) -> i32

  // COM: states with resets are not touched
  // CHECK-NEXT: [[V11:%.+]] = arc.state @Bar(%arg0) clock %clk lat 1 :
  %23 = arc.state @Bar(%arg0) clock %clk lat 1 : (i32) -> i32
  // CHECK-NEXT: [[V12:%.+]] = arc.state @Bar([[V11]]) clock %clk reset %rst lat 1 :
  %24 = arc.state @Bar(%23) clock %clk reset %rst lat 1 : (i32) -> i32
  // CHECK-NEXT: arc.state @Bar([[V12]]) clock %clk lat 1 :
  %25 = arc.state @Bar(%24) clock %clk lat 1 : (i32) -> i32

  // COM: using own result value
  // CHECK-NEXT: [[V13:%.+]] = arc.state @Bar([[V13]]) clock %clk lat 1 :
  %26 = arc.state @Bar(%26) clock %clk lat 1 : (i32) -> i32

  // COM: different clocks
  // CHECK-NEXT: arc.state @Bar(%arg0) clock %clk lat 1 :
  %27 = arc.state @Bar(%arg0) clock %clk lat 1 : (i32) -> i32
  // CHECK-NEXT: arc.state @Bar({{.*}}) clock %clk2 lat 1 :
  %28 = arc.state @Bar(%27) clock %clk2 lat 1 : (i32) -> i32

  // COM: can only partially take over latencies
  %29 = arc.state @Bar(%arg0) clock %clk lat 1 : (i32) -> i32
  // CHECK-NEXT: arc.state @Bar(%arg1) clock %clk lat 1 :
  %30 = arc.state @Bar(%arg1) clock %clk lat 2 : (i32) -> i32
  // CHECK-NEXT: arc.state @Baz(%arg0, {{.+}}) clock %clk lat 2 :
  %31:2 = arc.state @Baz(%29, %30) clock %clk lat 1 : (i32, i32) -> (i32, i32)

  // CHECK-NEXT: hw.output [[V0]], [[V3]]#0, [[V3]]#1, [[V5]]#0
  hw.output %2, %5#0, %5#1, %10#0, %12, %13, %16, %19, %22, %25, %28, %31 : i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32
}
arc.define @Bar(%arg0: i32) -> i32 {
  arc.output %arg0 : i32
}
arc.define @Baz(%arg0: i32, %arg1: i32) -> (i32, i32) {
  arc.output %arg1, %arg0 : i32, i32
}
