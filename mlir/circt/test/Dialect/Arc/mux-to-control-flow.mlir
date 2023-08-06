// RUN: circt-opt %s --arc-mux-to-control-flow | FileCheck %s

// CHECK-LABEL: @test1
arc.define @test1(%arg0: i41, %arg1: i1, %arg2: i1, %arg3: i1, %arg4: i1, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i1, %arg10: i1, %arg11: i1, %arg12: i1, %arg13: i1, %arg14: i1) -> i1 {
  %true = hw.constant true
  %0 = comb.extract %arg0 from 0 : (i41) -> i1
  %1 = comb.and bin %arg1, %0 : i1
  %2 = comb.and %arg2, %arg1, %arg3 : i1
  %3 = comb.or bin %1, %2 : i1
  %4 = comb.xor %3, %true : i1
  %5 = comb.and %arg4, %arg5 : i1
  %6 = comb.or %arg6, %5 : i1
  %7 = comb.and %arg7, %6 : i1
  // CHECK: [[ELSE:%.+]] = comb.mux bin %arg8, %arg5
  %8 = comb.mux bin %arg8, %arg5, %7 : i1
  %9 = comb.and %4, %8 : i1
  %10 = comb.xor %2, %true : i1
  %11 = comb.and %10, %8 : i1
  %12 = comb.mux bin %arg9, %9, %11 : i1
  %13 = comb.extract %arg0 from 19 : (i41) -> i1
  %14 = comb.xor bin %13, %true : i1
  %15 = comb.and bin %arg1, %14 : i1
  %16 = comb.xor %15, %true : i1
  %17 = comb.and %16, %8 : i1
  %18 = comb.and %arg10, %8 : i1
  %19 = comb.mux bin %arg11, %17, %18 : i1
  %20 = comb.mux bin %arg12, %12, %19 : i1
  %21 = comb.mux bin %arg13, %20, %8 : i1
  %22 = comb.and %arg14, %21 : i1
  arc.output %22 : i1
  // CHECK: [[IF:%.+]] = scf.if %arg13 -> (i1) {
  // CHECK: [[THEN:%.+]] = comb.mux bin %arg12
  // CHECK:   scf.yield [[THEN]] : i1
  // CHECK: } else {
  // CHECK:   scf.yield [[ELSE]] : i1
  // CHECK: }
  // CHECK: comb.and %arg14, [[IF]] : i1
}

// TODO: while above testcase should already provide high statement coverage,
// there are still cases that need to be tested. However, since the heuristics
// are not properly implemented yet, changing them would lead to breaking tests
// not because of incorrectness, but because of slightly different performance
// tradeoffs chosen. Bottom line is that it is hard to implement non-fragile tests
// in this FileCheck based setting before the heuristics are stable.
