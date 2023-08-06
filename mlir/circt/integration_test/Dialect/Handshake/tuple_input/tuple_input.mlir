// REQUIRES: iverilog,cocotb

// RUN: hlstool %s --dynamic-hw --ir-input-level 1 --buffering-strategy=cycles --verilog --lowering-options=disallowLocalVariables,disallowPackedStructAssignments > %t.sv && \
// RUN: circt-cocotb-driver.py --objdir=%T --topLevel=top --pythonModule=tuple_input --pythonFolder="%S,%S/.." %t.sv 2>&1 | FileCheck %s

// RUN: hlstool %s --dynamic-hw --ir-input-level 1 --buffering-strategy=all --verilog --lowering-options=disallowLocalVariables,disallowPackedStructAssignments > %t.sv && \
// RUN: circt-cocotb-driver.py --objdir=%T --topLevel=top --pythonModule=tuple_input --pythonFolder="%S,%S/.." %t.sv 2>&1 | FileCheck %s

// CHECK: ** TEST
// CHECK: ** TESTS=[[N:.*]] PASS=[[N]] FAIL=0 SKIP=0

module {
  handshake.func @top(%arg0: tuple<i32, i32>, %arg1: none, ...) -> (i32, none) attributes {argNames = ["in0", "inCtrl"], resNames = ["out0", "outCtrl"]} {
    %res:2 = handshake.unpack %arg0 : tuple<i32, i32>
    %sum = arith.addi %res#0, %res#1 : i32
    return %sum, %arg1 : i32, none
  }
}

