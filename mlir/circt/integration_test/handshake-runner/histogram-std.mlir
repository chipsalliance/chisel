// RUN: handshake-runner %s | FileCheck %s
// RUN: circt-opt -lower-std-to-handshake -handshake-materialize-forks-sinks %s | handshake-runner | FileCheck %s
// CHECK: 0

module {
  func.func @main() -> index {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %0 = memref.alloc() : memref<100xi32>
    %1 = memref.alloc() : memref<100xi32>
    %2 = memref.alloc() : memref<100xi32>
    cf.br ^bb1(%c0 : index)
  ^bb1(%3: index):	// 2 preds: ^bb0, ^bb2
    %4 = arith.cmpi slt, %3, %c10 : index
    cf.cond_br %4, ^bb2, ^bb3
  ^bb2:	// pred: ^bb1
    %5 = memref.load %0[%3] : memref<100xi32>
    %6 = memref.load %1[%3] : memref<100xi32>
    %7 = arith.index_cast %5 : i32 to index
    %8 = memref.load %2[%7] : memref<100xi32>
    %9 = arith.addi %8, %6 : i32
    memref.store %9, %2[%7] : memref<100xi32>
    %10 = arith.addi %3, %c1 : index
    cf.br ^bb1(%10 : index)
  ^bb3:	// pred: ^bb1
    return %c0 : index
  }
}
