// RUN: circt-opt -split-input-file --handshake-dematerialize-forks-sinks %s | FileCheck %s

// CHECK-LABEL:   handshake.func @gcd(
// CHECK-SAME:                        %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: none, ...) -> (i32, none)
// CHECK:           %[[VAL_3:.*]], %[[VAL_4:.*]] = control_merge %[[VAL_5:.*]], %[[VAL_6:.*]], %[[VAL_2]] : none, index
// CHECK:           %[[VAL_7:.*]] = mux %[[VAL_4]] {{\[}}%[[VAL_8:.*]], %[[VAL_9:.*]], %[[VAL_1]]] : index, i32
// CHECK:           %[[VAL_10:.*]] = mux %[[VAL_4]] {{\[}}%[[VAL_11:.*]], %[[VAL_12:.*]], %[[VAL_0]]] : index, i32
// CHECK:           %[[VAL_13:.*]] = arith.cmpi ne, %[[VAL_10]], %[[VAL_7]] : i32
// CHECK:           %[[VAL_14:.*]], %[[VAL_15:.*]] = cond_br %[[VAL_13]], %[[VAL_3]] : none
// CHECK:           %[[VAL_16:.*]], %[[VAL_17:.*]] = cond_br %[[VAL_13]], %[[VAL_7]] : i32
// CHECK:           %[[VAL_18:.*]], %[[VAL_19:.*]] = cond_br %[[VAL_13]], %[[VAL_10]] : i32
// CHECK:           %[[VAL_20:.*]] = arith.cmpi sgt, %[[VAL_18]], %[[VAL_16]] : i32
// CHECK:           %[[VAL_6]], %[[VAL_5]] = cond_br %[[VAL_20]], %[[VAL_14]] : none
// CHECK:           %[[VAL_21:.*]], %[[VAL_11]] = cond_br %[[VAL_20]], %[[VAL_18]] : i32
// CHECK:           %[[VAL_9]], %[[VAL_22:.*]] = cond_br %[[VAL_20]], %[[VAL_16]] : i32
// CHECK:           %[[VAL_12]] = arith.subi %[[VAL_21]], %[[VAL_9]] : i32
// CHECK:           %[[VAL_8]] = arith.subi %[[VAL_22]], %[[VAL_11]] : i32
// CHECK:           return %[[VAL_19]], %[[VAL_15]] : i32, none
// CHECK:         }
handshake.func @gcd(%arg0: i32, %arg1: i32, %arg2: none, ...) -> (i32, none) {
  %result, %index = control_merge %falseResult_5, %trueResult_4, %arg2 : none, index
  %0:2 = fork [2] %index : index
  %1 = mux %0#1 [%14, %11#0, %arg1] : index, i32
  %2:2 = fork [2] %1 : i32
  %3 = mux %0#0 [%13#0, %12, %arg0] : index, i32
  %4:2 = fork [2] %3 : i32
  %5 = arith.cmpi ne, %4#1, %2#1 : i32
  %6:3 = fork [3] %5 : i1
  %trueResult, %falseResult = cond_br %6#2, %result : none
  %trueResult_0, %falseResult_1 = cond_br %6#1, %2#0 : i32
  sink %falseResult_1 : i32
  %trueResult_2, %falseResult_3 = cond_br %6#0, %4#0 : i32
  %7:2 = fork [2] %trueResult_2 : i32
  %8:2 = fork [2] %trueResult_0 : i32
  %9 = arith.cmpi sgt, %7#1, %8#1 : i32
  %10:3 = fork [3] %9 : i1
  %trueResult_4, %falseResult_5 = cond_br %10#2, %trueResult : none
  %trueResult_6, %falseResult_7 = cond_br %10#1, %7#0 : i32
  %trueResult_8, %falseResult_9 = cond_br %10#0, %8#0 : i32
  %11:2 = fork [2] %trueResult_8 : i32
  %12 = arith.subi %trueResult_6, %11#1 : i32
  %13:2 = fork [2] %falseResult_7 : i32
  %14 = arith.subi %falseResult_9, %13#1 : i32
  return %falseResult_3, %falseResult : i32, none
}

// -----

// CHECK-LABEL:   handshake.func @external(
// CHECK-SAME:      i32, none, ...) -> none
handshake.func @external(%arg0: i32, %ctrl: none, ...) -> none
