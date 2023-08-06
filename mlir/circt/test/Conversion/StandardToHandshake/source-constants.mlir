// RUN: circt-opt -lower-std-to-handshake="source-constants" %s | FileCheck %s

// CHECK-LABEL:   handshake.func @foo(
// CHECK-SAME:                        %[[VAL_1:.*]]: none, ...) -> (i32, none)
// CHECK:           %[[VAL_1x:.*]] = merge %[[VAL_1]] : none
// CHECK:           %[[VAL_2:.*]] = source
// CHECK:           %[[VAL_3:.*]] = constant %[[VAL_2]] {value = 1 : i32} : i32
// CHECK:           return %[[VAL_3]], %[[VAL_1x]] : i32, none
// CHECK:         }

func.func @foo() -> i32 {
  %c1_i32 = arith.constant 1 : i32
  return %c1_i32 : i32
}
