// RUN: circt-opt --split-input-file -lower-std-to-handshake %s | FileCheck %s

// CHECK-LABEL: handshake.func @foo(
// CHECK-SAME:      %[[CTRL:.*]]: none, ...) -> none
// CHECK:         %[[CTRLX:.*]] = merge %[[CTRL]] : none
// CHECK:         return %[[CTRLX]] : none
func.func @foo() {
  return
}

// -----

// CHECK-LABEL: handshake.func @args(
// CHECK-SAME:      %[[ARG:.*]]: i32,
// CHECK-SAME:      %[[CTRL:.*]]: none, ...) -> (i32, none)
// CHECK:         %[[VAL:.*]] = merge %[[ARG]] : i32
// CHECK:         %[[CTRLX:.*]] = merge %[[CTRL]] : none
// CHECK:         return %[[VAL]], %[[CTRLX]] : i32, none
func.func @args(%a: i32) -> i32 {
  return %a: i32
}

// -----

// CHECK-LABEL: handshake.func private @ext(i32, i32, none, ...) -> (i32, none)
func.func private @ext(%a:i32, %b: i32) -> i32

