// RUN: circt-opt -lower-handshake-to-hw -split-input-file %s | FileCheck %s

// CHECK-LABEL:   hw.module @foo(
// CHECK-SAME:                   %[[VAL_0:.*]]: !esi.channel<i32>,
// CHECK-SAME:                   %[[VAL_1:.*]]: i1,
// CHECK-SAME:                   %[[VAL_2:.*]]: i1) -> (out0: !esi.channel<i32>) {
// CHECK:           hw.output %[[VAL_0]] : !esi.channel<i32>
// CHECK:         }

// CHECK-LABEL:   hw.module @bar(
// CHECK-SAME:                   %[[VAL_0:.*]]: !esi.channel<i32>,
// CHECK-SAME:                   %[[VAL_1:.*]]: i1,
// CHECK-SAME:                   %[[VAL_2:.*]]: i1) -> (out0: !esi.channel<i32>) {
// CHECK:           %[[VAL_3:.*]] = hw.instance "foo0" @foo(in: %[[VAL_0]]: !esi.channel<i32>, clock: %[[VAL_1]]: i1, reset: %[[VAL_2]]: i1) -> (out0: !esi.channel<i32>)
// CHECK:           hw.output %[[VAL_3]] : !esi.channel<i32>
// CHECK:         }
handshake.func @foo(%in : i32) -> (i32) {
    handshake.return %in : i32
}

handshake.func @bar(%in : i32) -> (i32) {
    %out = handshake.instance @foo(%in) : (i32) -> (i32)
    handshake.return %out : i32
}

// -----

// CHECK:         hw.module.extern @foo(%[[VAL_4:.*]]: !esi.channel<i32>, %[[VAL_5:.*]]: i1, %[[VAL_6:.*]]: i1) -> (out0: !esi.channel<i32>)

// CHECK-LABEL:   hw.module @bar(
// CHECK-SAME:                   %[[VAL_4]]: !esi.channel<i32>,
// CHECK-SAME:                   %[[VAL_5]]: i1,
// CHECK-SAME:                   %[[VAL_6]]: i1) -> (out0: !esi.channel<i32>) {
// CHECK:           %[[VAL_0:.*]] = hw.instance "foo0" @foo(in: %[[VAL_4]]: !esi.channel<i32>, clock: %[[VAL_5]]: i1, reset: %[[VAL_6]]: i1) -> (out0: !esi.channel<i32>)
// CHECK:           hw.output %[[VAL_0]] : !esi.channel<i32>
// CHECK:         }
handshake.func @foo(%in : i32) -> (i32)

handshake.func @bar(%in : i32) -> (i32) {
    %out = handshake.instance @foo(%in) : (i32) -> (i32)
    handshake.return %out : i32
}
