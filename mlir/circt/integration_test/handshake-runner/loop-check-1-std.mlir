// RUN: handshake-runner %s | FileCheck %s
// RUN: circt-opt -lower-std-to-handshake -handshake-materialize-forks-sinks %s | handshake-runner | FileCheck %s
// CHECK: 10

module {
  func.func @main() -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c5_i32 = arith.constant 5 : i32
    %0 = memref.alloc() : memref<64xi32>
    %1 = memref.alloc() : memref<64xi32>
    cf.br ^bb1(%c0 : index)
  ^bb1(%2: index):  // 2 preds: ^bb0, ^bb2
    %3 = arith.cmpi slt, %2, %c4 : index
    cf.cond_br %3, ^bb2, ^bb3
  ^bb2: // pred: ^bb1
    memref.store %c5_i32, %0[%2] : memref<64xi32>
    %4 = arith.addi %2, %c1 : index
    cf.br ^bb1(%4 : index)
  ^bb3: // pred: ^bb1
    cf.br ^bb4(%c0 : index)
  ^bb4(%5: index):  // 2 preds: ^bb3, ^bb5
    %6 = arith.cmpi slt, %5, %c4 : index
    cf.cond_br %6, ^bb5, ^bb6
  ^bb5: // pred: ^bb4
    %7 = memref.load %0[%5] : memref<64xi32>
    %8 = arith.addi %7, %7 : i32
    memref.store %8, %1[%5] : memref<64xi32>
    %9 = arith.addi %5, %c1 : index
    cf.br ^bb4(%9 : index)
  ^bb6: // pred: ^bb4
    %10 = memref.load %1[%c0] : memref<64xi32>
    return %10 : i32
  }
}
