// RUN: circt-opt -split-input-file %s | circt-opt | FileCheck %s

// CHECK-LABEL: handshake.func @unpack_pack(
// CHECK-SAME:                            %[[VAL_0:.*]]: tuple<i64, i32, i64>,
// CHECK-SAME:                            %[[VAL_1:.*]]: none, ...) -> (tuple<i32, i64>, none) attributes {argNames = ["in0", "inCtrl"], resNames = ["out0", "outCtrl"]} {
// CHECK:  %[[VAL_2:.*]]:3 = unpack %[[VAL_0]] : tuple<i64, i32, i64>
// CHECK:  %[[VAL_3:.*]] = arith.addi %[[VAL_2]]#0, %[[VAL_2]]#2 : i64
// CHECK:  %[[VAL_4:.*]] = pack %[[VAL_2]]#1, %[[VAL_3]] : tuple<i32, i64>
// CHECK:  return %[[VAL_4]], %[[VAL_1]] : tuple<i32, i64>, none
// CHECK:}

handshake.func @unpack_pack(%in: tuple<i64, i32, i64>, %arg1: none, ...) -> (tuple<i32, i64>, none) attributes {argNames = ["in0", "inCtrl"], resNames = ["out0", "outCtrl"]} {
  %a, %b, %c = handshake.unpack %in : tuple<i64,i32,i64>
  %sum = arith.addi %a, %c : i64

  %res = handshake.pack %b, %sum : tuple<i32, i64>

  return %res, %arg1 : tuple<i32, i64>, none
}

// -----

// CHECK-LABEL: handshake.func @with_attributes(
// CHECK-SAME:                                  %[[VAL_0:.*]]: tuple<i64, i32>,
// CHECK-SAME:                                  %[[VAL_1:.*]]: none, ...) -> (tuple<i64, i32>, none) attributes {argNames = ["in0", "inCtrl"], resNames = ["out0", "outCtrl"]} {
// CHECK:  %[[VAL_2:.*]]:2 = unpack %[[VAL_0]] {testAttr = "content"} : tuple<i64, i32>
// CHECK:  %[[VAL_3:.*]] = pack %[[VAL_2]]#0, %[[VAL_2]]#1 {testAttr2 = "content2", testAttr3 = "content3"} : tuple<i64, i32>
// CHECK:  return %[[VAL_3]], %[[VAL_1]] : tuple<i64, i32>, none
// CHECK:}

handshake.func @with_attributes(%in: tuple<i64, i32>, %arg1: none, ...) -> (tuple<i64, i32>, none) attributes {argNames = ["in0", "inCtrl"], resNames = ["out0", "outCtrl"]} {
  %a, %b = handshake.unpack %in {testAttr = "content"} : tuple<i64,i32>
  %res = handshake.pack %a, %b {testAttr2 = "content2", testAttr3 = "content3"} : tuple<i64, i32>

  return %res, %arg1 : tuple<i64, i32>, none
}
