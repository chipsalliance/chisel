// These tests will be only enabled if circt-lec is built.
// REQUIRES: circt-lec

// RUN: split-file %s %t

// Passing two input files
//  RUN: circt-lec %t/first.mlir %t/second.mlir -v=false | FileCheck %s
//  CHECK: c1 == c2

//--- first.mlir
hw.module @basic(%in: i1) -> (out: i1) {
  hw.output %in : i1
}

//--- second.mlir
hw.module @basic(%in: i1) -> (out: i1) {
  hw.output %in : i1
}
