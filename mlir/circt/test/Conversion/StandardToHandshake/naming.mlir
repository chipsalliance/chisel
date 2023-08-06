// RUN: circt-opt -lower-std-to-handshake %s | FileCheck %s

// CHECK-LABEL: handshake.func @main(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: none, ...) -> (i32, none) attributes {argNames = ["a", "b", "c", "in3"], resNames = ["res", "outCtrl"]} {
func.func @main(%arg0 : i32, %b : i32, %c: i32) -> i32 attributes {argNames = ["a", "b", "c"], resNames = ["res", "outCtrl"]} {
  return %arg0 : i32
}
