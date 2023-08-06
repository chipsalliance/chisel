// RUN: circt-opt -split-input-file %s | circt-opt | FileCheck %s

// CHECK-LABEL:   handshake.func private @private_func(
// CHECK-SAME:                   %[[VAL_0:.*]]: i32,
// CHECK-SAME:                   %[[VAL_1:.*]]: none, ...) -> (i32, none) attributes {argNames = ["arg0", "ctrl"], resNames = ["out0", "out1"]} {
// CHECK:           return %[[VAL_0]], %[[VAL_1]] : i32, none
// CHECK:         }
handshake.func private @private_func(%arg0 : i32, %ctrl: none) -> (i32, none) {
  return %arg0, %ctrl : i32, none
}

// -----

// CHECK-LABEL:   handshake.func public @public_func(
// CHECK-SAME:                   %[[VAL_0:.*]]: i32,
// CHECK-SAME:                   %[[VAL_1:.*]]: none, ...) -> (i32, none) attributes {argNames = ["arg0", "ctrl"], resNames = ["out0", "out1"]} {
// CHECK:           return %[[VAL_0]], %[[VAL_1]] : i32, none
// CHECK:         }
handshake.func public @public_func(%arg0 : i32, %ctrl: none) -> (i32, none) {
  return %arg0, %ctrl : i32, none
}

// -----

// CHECK-LABEL:   handshake.func public @no_none_type(
// CHECK-SAME:                   ...) attributes {argNames = [], resNames = []} {
// CHECK:           return
// CHECK:         }
handshake.func public @no_none_type() {
  return
}

// -----

// CHECK-LABEL:   handshake.func @external(
// CHECK-SAME:      i32, none, ...) -> none attributes {argNames = ["arg0", "ctrl"], resNames = ["out0"]}
handshake.func @external(%arg0: i32, %ctrl: none, ...) -> none

// ----

// CHECK-LABEL:   handshake.func @no_ssa_names(
// CHECK-SAME:      i32, none, ...) -> none attributes {argNames = ["in0", "in1"], resNames = ["out0"]}
handshake.func @no_ssa_names(i32, none, ...) -> none
