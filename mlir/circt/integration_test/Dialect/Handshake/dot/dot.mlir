// REQUIRES: iverilog,cocotb

// RUN: hlstool %s --dynamic-hw --buffering-strategy=cycles --verilog --lowering-options=disallowLocalVariables > %t.sv && \
// RUN: circt-cocotb-driver.py --objdir=%T --topLevel=top --pythonModule=dot --pythonFolder="%S,%S/.." %t.sv 2>&1 | FileCheck %s

// RUN: hlstool %s --dynamic-hw --buffering-strategy=all --verilog --lowering-options=disallowLocalVariables > %t.sv && \
// RUN: circt-cocotb-driver.py --objdir=%T --topLevel=top --pythonModule=dot --pythonFolder="%S,%S/.." %t.sv 2>&1 | FileCheck %s

// CHECK: ** TEST
// CHECK: ** TESTS=[[N:.*]] PASS=[[N]] FAIL=0 SKIP=0

module {
  func.func @top() -> i32 {
    %c123_i32 = arith.constant 123 : i32
    %c456_i32 = arith.constant 456 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<64xi32>
    %1 = memref.alloc() : memref<64xi32>
    cf.br ^bb1(%c0 : index)
  ^bb1(%2: index):  // 2 preds: ^bb0, ^bb2
    %3 = arith.cmpi slt, %2, %c64 : index
    cf.cond_br %3, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    memref.store %c123_i32, %0[%2] : memref<64xi32>
    memref.store %c456_i32, %1[%2] : memref<64xi32>
    %4 = arith.addi %2, %c1 : index
    cf.br ^bb1(%4 : index)
  ^bb3:  // pred: ^bb1
    cf.br ^bb4(%c0, %c0_i32 : index, i32)
  ^bb4(%5: index, %6: i32):  // 2 preds: ^bb3, ^bb5
    %7 = arith.cmpi slt, %5, %c64 : index
    cf.cond_br %7, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %8 = memref.load %0[%5] : memref<64xi32>
    %9 = memref.load %1[%5] : memref<64xi32>
    %10 = arith.muli %8, %9 : i32
    %11 = arith.addi %6, %10 : i32
    %12 = arith.addi %5, %c1 : index
    cf.br ^bb4(%12, %11 : index, i32)
  ^bb6:  // pred: ^bb4
    return %6 : i32
  }
}

