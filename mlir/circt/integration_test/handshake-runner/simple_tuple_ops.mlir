// RUN: handshake-runner %s "(64, 32, 64)" | FileCheck %s
// CHECK: (128, 32)

module {
  handshake.func @main(%arg0: tuple<i64, i32, i64>, %ctrl: none, ...) -> (tuple<i64, i32>, none) {
    %0, %1, %2 = unpack %arg0 : tuple<i64, i32, i64>
    %sum = arith.addi %0, %2 : i64
    %res = pack %sum, %1 : tuple<i64, i32>
    return %res, %ctrl : tuple<i64, i32>, none
  }
}
