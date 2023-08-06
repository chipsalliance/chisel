// REQUIRES: iverilog,cocotb

// RUN: hlstool %s --dynamic-hw --buffering-strategy=cycles --verilog --lowering-options=disallowLocalVariables > %t.sv && \
// RUN: circt-cocotb-driver.py --objdir=%T --topLevel=top --pythonModule=nested_loops --pythonFolder="%S,%S/.." %t.sv 2>&1 | FileCheck %s

// RUN: hlstool %s --dynamic-hw --buffering-strategy=all --verilog --lowering-options=disallowLocalVariables > %t.sv && \
// RUN: circt-cocotb-driver.py --objdir=%T --topLevel=top --pythonModule=nested_loops --pythonFolder="%S,%S/.." %t.sv 2>&1 | FileCheck %s

// Locking the circt should yield the same result

// RUN: hlstool %s --dynamic-hw --buffering-strategy=all --dynamic-parallelism=locking --verilog --lowering-options=disallowLocalVariables > %t.sv && \
// RUN: circt-cocotb-driver.py --objdir=%T --topLevel=top --pythonModule=nested_loops --pythonFolder="%S,%S/.." %t.sv 2>&1 | FileCheck %s

// CHECK: ** TEST
// CHECK: ** TESTS=[[NUM:.*]] PASS=[[NUM]] FAIL=0 SKIP=0

func.func @top(%N: i64) -> i64 {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  cf.br ^bb1(%c0, %c0: i64, i64)
^bb1(%i: i64, %acc: i64):
  %cond = arith.cmpi sle, %i, %N : i64
  cf.cond_br %cond, ^bb2(%c0, %c0: i64, i64), ^exit
^bb2(%j: i64, %innerAcc: i64):
  %cond1 = arith.cmpi sle, %j, %i : i64
  cf.cond_br %cond1, ^bb3, ^bb4
^bb3:
  %nia = arith.addi %j, %innerAcc : i64
  %nj = arith.addi %j, %c1 : i64
  cf.br ^bb2(%nj, %nia : i64, i64)
^bb4:
  %na = arith.addi %acc, %innerAcc : i64
  %ni = arith.addi %i, %c1 : i64
  cf.br ^bb1(%ni, %na : i64, i64)
^exit:
  return %acc: i64
}

