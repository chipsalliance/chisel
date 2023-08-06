// RUN: circt-opt --split-input-file --handshake-remove-buffers %s | FileCheck %s

// CHECK-LABEL:   handshake.func @simple_c(
// CHECK-SAME:                             %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: none, ...) -> (i32, none)
// CHECK:           %[[VAL_3:.*]] = arith.addi %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           return %[[VAL_3]], %[[VAL_2]] : i32, none
// CHECK:         }
handshake.func @simple_c(%arg0: i32, %arg1: i32, %arg2: none) -> (i32, none) {
  %0 = buffer [2] seq %arg0 : i32
  %1 = buffer [2] seq %arg1 : i32
  %2 = arith.addi %0, %1 : i32
  %3 = buffer [2] seq %2 : i32
  %4 = buffer [2] seq %arg2 : none
  return %3, %4 : i32, none
}

// -----

// CHECK-LABEL:   handshake.func @external(
// CHECK-SAME:      i32, none, ...) -> none
handshake.func @external(%arg0: i32, %ctrl: none, ...) -> none
