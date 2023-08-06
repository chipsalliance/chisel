// RUN: circt-opt -split-input-file %s | circt-opt | FileCheck %s

// CHECK-LABEL:   handshake.func @single_in(
// CHECK-SAME:                              %[[VAL_0:.*]]: none, ...) -> none
// CHECK:           %[[VAL_1:.*]] = sync %[[VAL_0]] : none
// CHECK:           return %[[VAL_1]] : none
// CHECK:         }

handshake.func @single_in(%in0: none) -> none {
  %ctrlOut = sync %in0 : none
  return %ctrlOut : none
}

// -----

// CHECK-LABEL:   handshake.func @multi_in(
// CHECK-SAME:                             %[[VAL_0:.*]]: none,
// CHECK-SAME:                             %[[VAL_1:.*]]: i32,
// CHECK-SAME:                             %[[VAL_2:.*]]: i512, ...) -> (none, i32, i512)
// CHECK:           %[[VAL_3:.*]]:3 = sync %[[VAL_0]], %[[VAL_1]], %[[VAL_2]] : none, i32, i512
// CHECK:           return %[[VAL_3]]#0, %[[VAL_3]]#1, %[[VAL_3]]#2 : none, i32, i512
// CHECK:         }

handshake.func @multi_in(%in0: none, %in1: i32, %in2: i512) -> (none, i32, i512) {
  %out:3 = sync %in0, %in1, %in2 : none, i32, i512
  return %out#0, %out#1, %out#2 : none, i32, i512
}

// -----

// CHECK-LABEL:   handshake.func @attr(
// CHECK-SAME:                         %[[VAL_0:.*]]: i64,
// CHECK-SAME:                         %[[VAL_1:.*]]: i32, ...) -> (i64, i32)
// CHECK:           %[[VAL_2:.*]]:2 = sync %[[VAL_0]], %[[VAL_1]] {test_attr = "attribute"} : i64, i32
// CHECK:           return %[[VAL_2]]#0, %[[VAL_2]]#1 : i64, i32
// CHECK:         }

handshake.func @attr(%in0: i64, %in1: i32) -> (i64, i32) {
  %out:2 = sync %in0, %in1 {test_attr = "attribute"} : i64, i32
  return %out#0, %out#1 : i64, i32
}
