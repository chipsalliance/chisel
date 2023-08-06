// RUN: handshake-runner %s 2,3,4,5 | FileCheck %s
// RUN: circt-opt -lower-std-to-handshake -handshake-materialize-forks-sinks %s | handshake-runner - 2,3,4,5 | FileCheck %s
// CHECK: 2 2,3,4,5

module {
  func.func @main(%0 : memref<4xi32>) -> i32{
    %c0 = arith.constant 0 : index
    %1 = memref.load %0[%c0] : memref<4xi32>
    return %1 : i32
    }

}
