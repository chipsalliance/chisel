// REQUIRES: iverilog,cocotb

// RUN: hlstool %s --sv-trace-iverilog --dynamic-hw --buffering-strategy=cycles --verilog --lowering-options=disallowLocalVariables > %t.sv && \
// RUN: circt-cocotb-driver.py --objdir=%T --topLevel=top --pythonModule=tp_memory --pythonFolder="%S,%S/.." %t.sv 2>&1 | FileCheck %s

// RUN: hlstool %s --sv-trace-iverilog --dynamic-hw --buffering-strategy=all --verilog --lowering-options=disallowLocalVariables > %t.sv && \
// RUN: circt-cocotb-driver.py --objdir=%T --topLevel=top --pythonModule=tp_memory --pythonFolder="%S,%S/.." %t.sv 2>&1 | FileCheck %s

// Locking the circt should yield the same result
// RUN: hlstool %s --dynamic-hw --buffering-strategy=all --dynamic-parallelism=locking --verilog --lowering-options=disallowLocalVariables > %t.sv && \
// RUN: circt-cocotb-driver.py --objdir=%T --topLevel=top --pythonModule=tp_memory --pythonFolder="%S,%S/.." %t.sv 2>&1 | FileCheck %s

// CHECK:      ** TEST
// CHECK-NEXT: ********************************
// CHECK-NEXT: ** tp_memory.oneInput
// CHECK-NEXT: ** tp_memory.multipleInputs
// CHECK-NEXT: ********************************
// CHECK-NEXT: ** TESTS=2 PASS=2 FAIL=0 SKIP=0
// CHECK-NEXT: ********************************

module {
  func.func @top(%val: i64, %write: i1) -> i64 {
    %mem = memref.alloc() : memref<1xi64>
    %0 = arith.constant 0 : index
    cf.cond_br %write, ^w, ^r
  ^w:
    memref.store %val, %mem[%0] : memref<1xi64>
    cf.br ^end(%val: i64)
  ^r:
    %r = memref.load %mem[%0] : memref<1xi64>
    cf.br ^end(%r: i64)
  ^end(%res: i64):
    return %res: i64
  }
}
