// REQUIRES: iverilog,cocotb

// RUN: circt-opt %s --insert-merge-blocks | \
// RUN: hlstool --dynamic-hw --buffering-strategy=cycles --verilog --lowering-options=disallowLocalVariables > %t.sv
// RUN: circt-cocotb-driver.py --objdir=%T --topLevel=top --pythonModule=nested_diamonds --pythonFolder="%S,%S/.." %t.sv 2>&1 | FileCheck %s

// @mortbopet: this is currently disabled due to deadlocking.
// RUN: circt-opt %s --insert-merge-blocks | \
// RUN: hlstool --dynamic-hw --buffering-strategy=all --verilog --lowering-options=disallowLocalVariables > %t.sv
// RUN: circt-cocotb-driver.py --objdir=%T --topLevel=top --pythonModule=nested_diamonds --pythonFolder="%S,%S/.." %t.sv 2>&1 | FileCheck %s

// Locking the circt should yield the same result
// RUN: hlstool %s --dynamic-hw --buffering-strategy=all --dynamic-parallelism=locking --verilog --lowering-options=disallowLocalVariables > %t.sv
// DISABLED: circt-cocotb-driver.py --objdir=%T --topLevel=top --pythonModule=nested_diamonds --pythonFolder="%S,%S/.." %t.sv 2>&1 | FileCheck %s

// CHECK: ** TEST
// CHECK: ** TESTS=[[NUM:.*]] PASS=[[NUM]] FAIL=0 SKIP=0

func.func @top(%i: i64) -> i64 {
  %c100 = arith.constant 100 : i64
  %cond0 = arith.cmpi sle, %i, %c100 : i64
  cf.cond_br %cond0, ^bb1, ^bb4
^bb1:
  %c50 = arith.constant 50 : i64
  %cond1 = arith.cmpi sle, %i, %c50 : i64
  cf.cond_br %cond1, ^bb2, ^bb3
^bb2:
  %c0 = arith.constant 0 : i64
  cf.br ^exit(%c0: i64)
^bb3:
  %c1 = arith.constant 1 : i64
  cf.br ^exit(%c1: i64)
^bb4:
  %c200 = arith.constant 200 : i64
  %cond2 = arith.cmpi sle, %i, %c200 : i64
  cf.cond_br %cond2, ^bb5, ^bb6
^bb5:
  %c2 = arith.constant 2 : i64
  cf.br ^exit(%c2: i64)
^bb6:
  %c3 = arith.constant 3 : i64
  cf.br ^exit(%c3: i64)
^exit(%res: i64):
  return %res: i64
}

