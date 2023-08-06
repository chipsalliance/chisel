// RUN: circt-opt --split-input-file -handshake-lock-functions %s | FileCheck %s

// CHECK-LABEL:   handshake.func @single_block(
// CHECK-SAME:        %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32,
// CHECK-SAME:        %[[VAL_2:.*]]: none, ...) -> (i32, none)
// CHECK:           %[[VAL_3:.*]] = buffer [1] seq %[[VAL_4:.*]] {initValues = [0]} : none
// CHECK:           %[[VAL_5:.*]]:4 = sync %[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]] : i32, i32, none, none
// CHECK:           %[[VAL_6:.*]] = arith.addi %[[VAL_5]]#0, %[[VAL_5]]#1 : i32
// CHECK:           %[[VAL_4]] = join %[[VAL_6]], %[[VAL_5]]#2, %[[VAL_5]]#3 : i32, none, none
// CHECK:           return %[[VAL_6]], %[[VAL_5]]#2 : i32, none
// CHECK:         }

handshake.func @single_block(%arg0: i32, %arg1: i32, %arg2: none, ...) -> (i32, none) {
  %0 = arith.addi %arg0, %arg1 : i32
  return %0, %arg2 : i32, none
}

// -----

// CHECK-LABEL:   handshake.func @triangle(
// CHECK-SAME:                             %[[VAL_0:.*]]: i32,
// CHECK-SAME:                             %[[VAL_1:.*]]: i1,
// CHECK-SAME:                             %[[VAL_2:.*]]: none, ...) -> (i32, none)
// CHECK:           %[[VAL_3:.*]] = buffer [1] seq %[[VAL_4:.*]] {initValues = [0]} : none
// CHECK:           %[[VAL_5:.*]]:4 = sync %[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]] : i32, i1, none, none
// CHECK:           %[[VAL_6:.*]]:2 = fork [2] %[[VAL_5]]#1 : i1
// CHECK:           %[[VAL_7:.*]], %[[VAL_8:.*]] = cond_br %[[VAL_6]]#1, %[[VAL_5]]#0 : i32
// CHECK:           sink %[[VAL_7]] : i32
// CHECK:           %[[VAL_9:.*]], %[[VAL_10:.*]] = cond_br %[[VAL_6]]#0, %[[VAL_5]]#2 : none
// CHECK:           %[[VAL_11:.*]]:2 = fork [2] %[[VAL_9]] : none
// CHECK:           %[[VAL_12:.*]] = constant %[[VAL_11]]#0 {value = 42 : i32} : i32
// CHECK:           %[[VAL_13:.*]], %[[VAL_14:.*]] = control_merge %[[VAL_11]]#1, %[[VAL_10]] : none, index
// CHECK:           %[[VAL_15:.*]] = mux %[[VAL_14]] {{\[}}%[[VAL_12]], %[[VAL_8]]] : index, i32
// CHECK:           %[[VAL_4]] = join %[[VAL_15]], %[[VAL_13]], %[[VAL_5]]#3 : i32, none, none
// CHECK:           return %[[VAL_15]], %[[VAL_13]] : i32, none
// CHECK:         }

handshake.func @triangle(%arg0: i32, %arg1: i1, %arg2: none, ...) -> (i32, none) {
  %0:2 = fork [2] %arg1 : i1
  %trueResult, %falseResult = cond_br %0#1, %arg0 : i32
  sink %trueResult : i32
  %trueResult_0, %falseResult_1 = cond_br %0#0, %arg2 : none
  %1:2 = fork [2] %trueResult_0 : none
  %2 = constant %1#0 {value = 42 : i32} : i32
  %result, %index = control_merge %1#1, %falseResult_1 : none, index
  %3 = mux %index [%2, %falseResult] : index, i32
  return %3, %result : i32, none
}

// -----

// CHECK-LABEL:   handshake.func @external(
// CHECK-SAME:      i32, none, ...) -> none
handshake.func @external(%arg0: i32, %ctrl: none, ...) -> none
