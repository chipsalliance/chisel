// RUN: circt-opt -lower-std-to-handshake -split-input-file %s | FileCheck %s

// CHECK-LABEL:   handshake.func @bar(
// CHECK-SAME:                        %[[VAL_0:.*]]: i32,
// CHECK-SAME:                        %[[VAL_1:.*]]: none, ...) -> (i32, none)
// CHECK:           %[[VAL_2:.*]] = merge %[[VAL_0]] : i32
// CHECK:           %[[VAL_1x:.*]] = merge %[[VAL_1]] : none
// CHECK:           return %[[VAL_2]], %[[VAL_1x]] : i32, none
// CHECK:         }
func.func @bar(%0 : i32) -> i32 {
  return %0 : i32
}

// CHECK-LABEL:   handshake.func @foo(
// CHECK-SAME:                        %[[VAL_0:.*]]: i32,
// CHECK-SAME:                        %[[VAL_1:.*]]: none, ...) -> (i32, none)
// CHECK:           %[[VAL_2:.*]] = merge %[[VAL_0]] : i32
// CHECK:           %[[VAL_1x:.*]] = merge %[[VAL_1]] : none
// CHECK:           %[[VAL_3:.*]]:2 = instance @bar(%[[VAL_2]], %[[VAL_1x]]) : (i32, none) -> (i32, none)
// CHECK:           return %[[VAL_3]]#0, %[[VAL_1x]] : i32, none
// CHECK:         }
func.func @foo(%0 : i32) -> i32 {
  %a1 = call @bar(%0) : (i32) -> i32
  return %a1 : i32
}

// -----

// Branching control flow with calls in each branch.

// CHECK-LABEL:   handshake.func @add(
func.func @add(%arg0 : i32, %arg1: i32) -> i32 {
  %0 = arith.addi %arg0, %arg1 : i32
  return %0 : i32
}

// CHECK-LABEL:   handshake.func @sub(
func.func @sub(%arg0 : i32, %arg1: i32) -> i32 {
  %0 = arith.subi %arg0, %arg1 : i32
  return %0 : i32
}

// CHECK:   handshake.func @main(%[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: none, ...) -> (i32, none)
// CHECK:           %[[VAL_4:.*]] = merge %[[VAL_0]] : i32
// CHECK:           %[[VAL_5:.*]] = merge %[[VAL_1]] : i32
// CHECK:           %[[VAL_6:.*]] = merge %[[VAL_2]] : i1
// CHECK:           %[[VAL_7:.*]] = buffer [2] fifo %[[VAL_6]] : i1
// CHECK:           %[[VAL_3x:.*]] = merge %[[VAL_3]] : none
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]] = cond_br %[[VAL_6]], %[[VAL_4]] : i32
// CHECK:           %[[VAL_10:.*]], %[[VAL_11:.*]] = cond_br %[[VAL_6]], %[[VAL_5]] : i32
// CHECK:           %[[VAL_12:.*]], %[[VAL_13:.*]] = cond_br %[[VAL_6]], %[[VAL_3x]] : none
// CHECK:           %[[VAL_14:.*]] = merge %[[VAL_8]] : i32
// CHECK:           %[[VAL_15:.*]] = merge %[[VAL_10]] : i32
// CHECK:           %[[VAL_16:.*]], %[[VAL_17:.*]] = control_merge %[[VAL_12]] : none, index
// CHECK:           %[[VAL_18:.*]]:2 = instance @add(%[[VAL_14]], %[[VAL_15]], %[[VAL_16]]) : (i32, i32, none) -> (i32, none)
// CHECK:           %[[VAL_19:.*]] = br %[[VAL_16]] : none
// CHECK:           %[[VAL_20:.*]] = br %[[VAL_18]]#0 : i32
// CHECK:           %[[VAL_21:.*]] = merge %[[VAL_9]] : i32
// CHECK:           %[[VAL_22:.*]] = merge %[[VAL_11]] : i32
// CHECK:           %[[VAL_23:.*]], %[[VAL_24:.*]] = control_merge %[[VAL_13]] : none, index
// CHECK:           %[[VAL_25:.*]]:2 = instance @sub(%[[VAL_21]], %[[VAL_22]], %[[VAL_23]]) : (i32, i32, none) -> (i32, none)
// CHECK:           %[[VAL_26:.*]] = br %[[VAL_23]] : none
// CHECK:           %[[VAL_27:.*]] = br %[[VAL_25]]#0 : i32
// CHECK:           %[[VAL_30:.*]] = mux %[[VAL_29:.*]] {{\[}}%[[VAL_27]], %[[VAL_20]]] : index, i32
// CHECK:           %[[VAL_28:.*]] = mux %[[VAL_7:.*]] {{\[}}%[[VAL_26]], %[[VAL_19]]] : i1, none
// CHECK:           %[[VAL_29]] = arith.index_cast %[[VAL_7]] : i1 to index
// CHECK:           return %[[VAL_30]], %[[VAL_28]] : i32, none
// CHECK:         }
func.func @main(%arg0 : i32, %arg1 : i32, %cond : i1) -> i32 {
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  %0 = call @add(%arg0, %arg1) : (i32, i32) -> i32
  cf.br ^bb3(%0 : i32)
^bb2:
  %1 = call @sub(%arg0, %arg1) : (i32, i32) -> i32
  cf.br ^bb3(%1 : i32)
^bb3(%res : i32):
  return %res : i32
}
