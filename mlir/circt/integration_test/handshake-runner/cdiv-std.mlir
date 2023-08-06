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
    %3 = memref.alloc() : memref<100xi32>
    %4 = memref.alloc() : memref<100xi32>
    %5 = memref.alloc() : memref<100xi32>
    %6 = memref.alloc() : memref<100xi32>
    %7 = memref.alloc() : memref<100xi32>
    %8 = memref.alloc() : memref<100xi32>
    %9 = memref.alloc() : memref<100xi32>
    %10 = memref.alloc() : memref<100xi1>
    cf.br ^bb1(%c0 : index)
  ^bb1(%11: index):	// 2 preds: ^bb0, ^bb5
    %12 = arith.cmpi slt, %11, %c10 : index
    cf.cond_br %12, ^bb2, ^bb6
  ^bb2:	// pred: ^bb1
    %13 = memref.load %0[%11] : memref<100xi32>
    %14 = memref.load %2[%11] : memref<100xi32>
    %15 = memref.load %0[%11] : memref<100xi32>
    %16 = memref.load %2[%11] : memref<100xi32>
    %17 = memref.load %10[%11] : memref<100xi1>
    cf.cond_br %17, ^bb3, ^bb4
  ^bb3:	// pred: ^bb2
    %18 = memref.load %6[%11] : memref<100xi32>
    %19 = memref.load %9[%11] : memref<100xi32>
    %20 = arith.muli %13, %18 : i32
    %21 = arith.muli %15, %18 : i32
    %22 = arith.addi %20, %15 : i32
    %23 = arith.subi %13, %21 : i32
    %24 = arith.muli %22, %19 : i32
    %25 = arith.muli %23, %19 : i32
    memref.store %24, %5[%11] : memref<100xi32>
    memref.store %25, %4[%11] : memref<100xi32>
    cf.br ^bb5
  ^bb4:	// pred: ^bb2
    %26 = memref.load %7[%11] : memref<100xi32>
    %27 = memref.load %8[%11] : memref<100xi32>
    %28 = arith.muli %13, %26 : i32
    %29 = arith.muli %15, %26 : i32
    %30 = arith.addi %29, %13 : i32
    %31 = arith.subi %28, %15 : i32
    %32 = arith.muli %30, %27 : i32
    %33 = arith.muli %31, %27 : i32
    memref.store %32, %5[%11] : memref<100xi32>
    memref.store %33, %4[%11] : memref<100xi32>
    cf.br ^bb5
  ^bb5:	// 2 preds: ^bb3, ^bb4
    %34 = arith.addi %11, %c1 : index
    cf.br ^bb1(%34 : index)
  ^bb6:	// pred: ^bb1
    return %c0 : index
  }
}
