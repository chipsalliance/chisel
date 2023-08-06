// Tests whether buffer strategies compose; this is done via. strategy 'allFIFO'
// which uses sequential buffers on cycles and FIFOs on everything else
// RUN: circt-opt --handshake-insert-buffers="strategy=allFIFO" %s | FileCheck %s

// CHECK-LABEL:   handshake.func @foo(
// CHECK-SAME:                        %[[VAL_0:.*]]: i32,
// CHECK-SAME:                        %[[VAL_1:.*]]: none, ...) -> (i32, none)
// CHECK:           %[[VAL_2:.*]] = buffer [2] fifo %[[VAL_1]] : none
// CHECK:           %[[VAL_3:.*]] = buffer [2] fifo %[[VAL_0]] : i32
// CHECK:           %[[VAL_4:.*]]:2 = fork [2] %[[VAL_3]] : i32
// CHECK:           %[[VAL_5:.*]] = buffer [2] fifo %[[VAL_4]]#1 : i32
// CHECK:           %[[VAL_6:.*]] = buffer [2] fifo %[[VAL_4]]#0 : i32
// CHECK:           %[[VAL_7:.*]] = mux %[[VAL_6]] {{\[}}%[[VAL_5]], %[[VAL_8:.*]]] : i32, i32
// CHECK:           %[[VAL_9:.*]] = buffer [2] seq %[[VAL_7]] : i32
// CHECK:           %[[VAL_10:.*]]:2 = fork [2] %[[VAL_9]] : i32
// CHECK:           %[[VAL_11:.*]] = buffer [2] fifo %[[VAL_10]]#1 : i32
// CHECK:           %[[VAL_8]] = buffer [2] fifo %[[VAL_10]]#0 : i32
// CHECK:           return %[[VAL_11]], %[[VAL_2]] : i32, none
// CHECK:         }

handshake.func @foo(%arg0 : i32, %ctrl : none) -> (i32, none) {
  %0:2 = fork [2] %arg0 : i32
  %1 = mux %0#0 [%0#1, %2#0] : i32, i32
  %2:2 = fork [2] %1 : i32
  return %2#1, %ctrl : i32, none
}

// -----

// CHECK-LABEL:   handshake.func @external(
// CHECK-SAME:      i32, none, ...) -> none
handshake.func @external(%arg0: i32, %ctrl: none, ...) -> none
