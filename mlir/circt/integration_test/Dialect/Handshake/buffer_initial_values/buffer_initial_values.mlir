// REQUIRES: iverilog,cocotb

// RUN: hlstool %s --dynamic-hw --buffering-strategy=cycles --verilog --lowering-options=disallowLocalVariables > %t.sv && \
// RUN: circt-cocotb-driver.py --objdir=%T --topLevel=top --pythonModule=buffer_initial_values --pythonFolder="%S,%S/.." %t.sv 2>&1 | FileCheck %s

// RUN: hlstool %s --dynamic-hw --buffering-strategy=all --verilog --lowering-options=disallowLocalVariables > %t.sv && \
// RUN: circt-cocotb-driver.py --objdir=%T --topLevel=top --pythonModule=buffer_initial_values --pythonFolder="%S,%S/.." %t.sv 2>&1 | FileCheck %s

// CHECK: ** TEST
// CHECK: ** TESTS=[[N:.*]] PASS=[[N]] FAIL=0 SKIP=0

module {
  handshake.func @top(%arg0: none) -> (i32, none) attributes {argNames = ["inCtrl"], resNames = ["out0", "outCtrl"]} {
    %d = buffer [2] seq %dTrue {initValues = [20, 10]} : i32
    %ctrl = merge %arg0, %cTrue : none

    %6 = constant %ctrl {value = 10 : i32} : i32
    %cond = arith.cmpi ne, %d, %6 : i32

    %dTrue, %dFalse = cond_br %cond, %d : i32
    %cTrue, %cFalse = cond_br %cond, %ctrl : none

    return %dFalse, %cFalse : i32, none
  }
}
