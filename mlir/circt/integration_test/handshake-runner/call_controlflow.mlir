// RUN: handshake-runner %s 3 2 1 | FileCheck %s
// RUN: circt-opt -lower-std-to-handshake -handshake-materialize-forks-sinks %s > handshake.mlir
// RUN  handshake-runner handshake.mlir 3 2 1 | FileCheck %s
// CHECK: 5

func.func @add(%arg0 : i32, %arg1: i32) -> i32 {
  %0 = arith.addi %arg0, %arg1 : i32
  return %0 : i32
}

func.func @sub(%arg0 : i32, %arg1: i32) -> i32 {
  %0 = arith.subi %arg0, %arg1 : i32
  return %0 : i32
}

func.func @main(%arg0 : i32, %arg1 : i32, %cond : i1) -> i32 {
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  %0 = call @add(%arg0, %arg1) : (i32, i32) -> i32
  cf.br ^bb3(%0 : i32)
^bb2:
  %1 = call @sub(%arg0, %arg1) : (i32, i32) -> i32
  cf.br ^bb3(%1 : i32)
^bb3(%res : i32):
  return %res : i32
}
