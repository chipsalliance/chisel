// RUN: handshake-runner %s | FileCheck %s
// RUN: circt-opt -lower-std-to-handshake -handshake-materialize-forks-sinks %s | handshake-runner | FileCheck %s
// CHECK: 0

module {
  func.func @main() -> index {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = memref.alloc() : memref<256xi32>
    %1 = memref.alloc() : memref<256xi32>
    %2 = memref.alloc() : memref<256xi32>
    %3 = memref.alloc() : memref<1xi32>
    cf.br ^bb1(%c0 : index)
  ^bb1(%4: index):	// 2 preds: ^bb0, ^bb8
    %5 = arith.cmpi slt, %4, %c4 : index
    cf.cond_br %5, ^bb2, ^bb9
  ^bb2:	// pred: ^bb1
    cf.br ^bb3(%c0 : index)
  ^bb3(%6: index):	// 2 preds: ^bb2, ^bb7
    %7 = arith.cmpi slt, %6, %c4 : index
    cf.cond_br %7, ^bb4, ^bb8
  ^bb4:	// pred: ^bb3
    memref.store %c0_i32, %3[%c0] : memref<1xi32>
    %8 = arith.muli %4, %c4 : index
    cf.br ^bb5(%c0 : index)
  ^bb5(%9: index):	// 2 preds: ^bb4, ^bb6
    %10 = arith.cmpi slt, %9, %c4 : index
    cf.cond_br %10, ^bb6, ^bb7
  ^bb6:	// pred: ^bb5
    %11 = arith.muli %9, %c4 : index
    %12 = arith.addi %8, %9 : index
    %13 = arith.addi %11, %6 : index
    %14 = memref.load %0[%12] : memref<256xi32>
    %15 = memref.load %1[%13] : memref<256xi32>
    %16 = memref.load %3[%c0] : memref<1xi32>
    %17 = arith.muli %14, %15 : i32
    %18 = arith.addi %16, %17 : i32
    memref.store %18, %3[%c0] : memref<1xi32>
    %19 = arith.addi %9, %c1 : index
    cf.br ^bb5(%19 : index)
  ^bb7:	// pred: ^bb5
    %20 = arith.addi %8, %6 : index
    %21 = memref.load %3[%c0] : memref<1xi32>
    memref.store %21, %2[%20] : memref<256xi32>
    %22 = arith.addi %6, %c1 : index
    cf.br ^bb3(%22 : index)
  ^bb8:	// pred: ^bb3
    %23 = arith.addi %4, %c1 : index
    cf.br ^bb1(%23 : index)
  ^bb9:	// pred: ^bb1
    return %c0 : index
  }
}
