// RUN: circt-opt %s --arc-isolate-clocks --verify-diagnostics

hw.module @m1(%cond: i1, %arg0: i32, %arg1: i32) -> (out: i32) {
  // expected-error @+1 {{operations with regions not supported yet!}}
  %0 = scf.if %cond -> i32 {
    scf.yield %arg0 : i32
  } else {
    scf.yield %arg1 : i32
  }
  hw.output %0 : i32
}
