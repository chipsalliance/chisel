// RUN: circt-opt %s --split-input-file | FileCheck %s


// CHECK-LABEL:   handshake.func @foo(...)
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   handshake.func @invalid_instance_op(
// CHECK-SAME:                                        %[[VAL_0:.*]]: i32, ...) -> i32
// CHECK:           instance @foo() : () -> ()
// CHECK:           return %[[VAL_0]] : i32
// CHECK:         }
handshake.func @foo() {
  return
}

handshake.func @invalid_instance_op(%arg0 : i32) -> i32 {
  instance @foo() : () -> ()
  return %arg0 : i32
}

// -----

// CHECK-LABEL:   handshake.func @foo(
// CHECK-SAME:                        %[[VAL_0:.*]]: i32, ...) -> i32
// CHECK:           return %[[VAL_0]] : i32
// CHECK:         }

// CHECK-LABEL:   handshake.func @invalid_instance_op(
// CHECK-SAME:                                        %[[VAL_0:.*]]: i32, ...) -> i32
// CHECK:           %[[VAL_1:.*]] = instance @foo(%[[VAL_0]]) : (i32) -> i32
// CHECK:           return %[[VAL_0]] : i32
// CHECK:         }

handshake.func @foo(%ctrl : i32) -> i32 {
  return %ctrl : i32
}

handshake.func @invalid_instance_op(%arg0 : i32) -> i32 {
  instance @foo(%arg0) : (i32) -> (i32)
  return %arg0 : i32
}
