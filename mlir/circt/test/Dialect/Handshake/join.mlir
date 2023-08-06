// RUN: circt-opt -split-input-file %s | circt-opt | FileCheck %s

// CHECK-LABEL:   handshake.func @simple_multi_input(
// CHECK-SAME:        %[[VAL_0:.*]]: none, %[[VAL_1:.*]]: none, %[[VAL_2:.*]]: none, ...) -> none
// CHECK:           %[[VAL_3:.*]] = join %[[VAL_0]], %[[VAL_1]], %[[VAL_2]] : none, none, none
// CHECK:           return %[[VAL_3]] : none
// CHECK:         }

handshake.func @simple_multi_input(%in0: none, %in1: none, %in2: none, ...) -> (none) {
  %ctrlOut = join %in0, %in1, %in2 : none, none, none
  return %ctrlOut : none
}

// -----

// CHECK-LABEL:   handshake.func @different_in_types(
// CHECK-SAME:                                       %[[VAL_0:.*]]: tuple<i64, i32, i64>,
// CHECK-SAME:                                       %[[VAL_1:.*]]: none, ...) -> none
// CHECK:           %[[VAL_2:.*]] = join %[[VAL_0]], %[[VAL_1]] : tuple<i64, i32, i64>, none
// CHECK:           return %[[VAL_2]] : none
// CHECK:         }

handshake.func @different_in_types(%in: tuple<i64, i32, i64>, %arg1: none, ...) -> (none) {
  %ctrlOut = join %in, %arg1 : tuple<i64, i32, i64>, none
  return %ctrlOut : none
}

// -----

// CHECK-LABEL:   handshake.func @superfluous_ctrl_attr(
// CHECK-SAME:        %[[VAL_0:.*]]: none, %[[VAL_1:.*]]: none, ...) -> none
// CHECK:           %[[VAL_2:.*]] = join %[[VAL_0]], %[[VAL_1]] : none, none
// CHECK:           return %[[VAL_2]] : none
// CHECK:         }

handshake.func @superfluous_ctrl_attr(%in: none, %arg1: none, ...) -> (none) {
  %ctrlOut = join %in, %arg1 {control = true} : none, none
  return %ctrlOut : none
}
