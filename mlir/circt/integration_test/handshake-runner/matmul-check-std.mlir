// RUN: handshake-runner %s | FileCheck %s
// RUN: circt-opt -lower-std-to-handshake -handshake-materialize-forks-sinks %s | handshake-runner | FileCheck %s
// CHECK: 200



module {
  func.func @main() -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c63 = arith.constant 63 : index
    %c64 = arith.constant 64 : index
    %c0_i32 = arith.constant 0 : i32
    %c5_i32 = arith.constant 5 : i32
    %0 = memref.alloc() : memref<64xi32>
    %1 = memref.alloc() : memref<64xi32>
    %2 = memref.alloc() : memref<64xi32>
    %3 = memref.alloc() : memref<1xi32>
    cf.br ^bb1(%c0 : index)
  ^bb1(%4: index):  // 2 preds: ^bb0, ^bb2
    %5 = arith.cmpi slt, %4, %c64 : index
    cf.cond_br %5, ^bb2, ^bb3
  ^bb2: // pred: ^bb1
    memref.store %c5_i32, %0[%4] : memref<64xi32>
    memref.store %c5_i32, %1[%4] : memref<64xi32>
    %6 = arith.addi %4, %c1 : index
    cf.br ^bb1(%6 : index)
  ^bb3: // pred: ^bb1
    cf.br ^bb4(%c0 : index)
  ^bb4(%7: index):  // 2 preds: ^bb3, ^bb11
    %8 = arith.cmpi slt, %7, %c8 : index
    cf.cond_br %8, ^bb5, ^bb12
  ^bb5: // pred: ^bb4
    cf.br ^bb6(%c0 : index)
  ^bb6(%9: index):  // 2 preds: ^bb5, ^bb10
    %10 = arith.cmpi slt, %9, %c8 : index
    cf.cond_br %10, ^bb7, ^bb11
  ^bb7: // pred: ^bb6
    memref.store %c0_i32, %3[%c0] : memref<1xi32>
    %11 = arith.muli %7, %c8 : index
    cf.br ^bb8(%c0 : index)
  ^bb8(%12: index): // 2 preds: ^bb7, ^bb9
    %13 = arith.cmpi slt, %12, %c8 : index
    cf.cond_br %13, ^bb9, ^bb10
  ^bb9: // pred: ^bb8
    %14 = arith.addi %11, %12 : index
    %15 = memref.load %0[%14] : memref<64xi32>
    %16 = arith.muli %12, %c8 : index
    %17 = arith.addi %16, %9 : index
    %18 = memref.load %1[%17] : memref<64xi32>
    %19 = memref.load %3[%c0] : memref<1xi32>
    %20 = arith.muli %15, %18 : i32
    %21 = arith.addi %19, %20 : i32
    memref.store %21, %3[%c0] : memref<1xi32>
    %22 = arith.addi %12, %c1 : index
    cf.br ^bb8(%22 : index)
  ^bb10:  // pred: ^bb8
    %23 = arith.addi %11, %9 : index
    %24 = memref.load %3[%c0] : memref<1xi32>
    memref.store %24, %2[%23] : memref<64xi32>
    %25 = arith.addi %9, %c1 : index
    cf.br ^bb6(%25 : index)
  ^bb11:  // pred: ^bb6
    %26 = arith.addi %7, %c1 : index
    cf.br ^bb4(%26 : index)
  ^bb12:  // pred: ^bb4
    %27 = memref.load %2[%c63] : memref<64xi32>
    return %27 : i32
  }
}
