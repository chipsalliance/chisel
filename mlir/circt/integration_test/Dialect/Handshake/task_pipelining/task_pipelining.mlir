// REQUIRES: iverilog,cocotb

// RUN: hlstool %s --dynamic-hw --buffering-strategy=cycles --verilog --lowering-options=disallowLocalVariables > %t.sv && \
// RUN: circt-cocotb-driver.py --objdir=%T --topLevel=top --pythonModule=task_pipelining --pythonFolder="%S,%S/.." %t.sv 2>&1 | FileCheck %s

// RUN: hlstool %s --dynamic-hw --buffering-strategy=all --verilog --lowering-options=disallowLocalVariables > %t.sv && \
// RUN: circt-cocotb-driver.py --objdir=%T --topLevel=top --pythonModule=task_pipelining --pythonFolder="%S,%S/.." %t.sv 2>&1 | FileCheck %s

// Locking the circt should yield the same result
// RUN: hlstool %s --dynamic-hw --buffering-strategy=all --dynamic-parallelism=locking --verilog --lowering-options=disallowLocalVariables > %t.sv && \
// RUN: circt-cocotb-driver.py --objdir=%T --topLevel=top --pythonModule=task_pipelining --pythonFolder="%S,%S/.." %t.sv 2>&1 | FileCheck %s

// CHECK:      ** TEST
// CHECK-NEXT: ********************************
// CHECK-NEXT: ** task_pipelining.oneInput
// CHECK-NEXT: ** task_pipelining.sendMultiple
// CHECK-NEXT: ********************************
// CHECK-NEXT: ** TESTS=2 PASS=2 FAIL=0 SKIP=0
// CHECK-NEXT: ********************************

module {
  func.func @top(%val: i64) -> i64 {
    %c0 = arith.constant 0 : i64
    %cond = arith.cmpi eq, %c0, %val : i64
    cf.cond_br %cond, ^1(%c0: i64), ^2(%val: i64)
  ^1(%i: i64):
    %c10 = arith.constant 100 : i64
    %lcond = arith.cmpi eq, %c10, %i : i64
    cf.cond_br %lcond, ^2(%i: i64), ^body
  ^body:
    %c1 = arith.constant 1 : i64
    %ni = arith.addi %i, %c1 : i64
    cf.br ^1(%ni: i64)
  ^2(%res: i64):
    return %res : i64
  }
}

