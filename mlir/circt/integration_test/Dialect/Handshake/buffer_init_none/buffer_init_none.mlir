// REQUIRES: iverilog,cocotb

// RUN: hlstool %s --dynamic-hw --buffering-strategy=cycles --verilog --lowering-options=disallowLocalVariables > %t.sv && \
// RUN: circt-cocotb-driver.py --objdir=%T --topLevel=top --pythonModule=buffer_init_none --pythonFolder="%S,%S/.." %t.sv 2>&1 | FileCheck %s

// RUN: hlstool %s --dynamic-hw --buffering-strategy=all --verilog --lowering-options=disallowLocalVariables > %t.sv && \
// RUN: circt-cocotb-driver.py --objdir=%T --topLevel=top --pythonModule=buffer_init_none --pythonFolder="%S,%S/.." %t.sv 2>&1 | FileCheck %s

// CHECK: ** TEST
// CHECK: ** TESTS=[[N:.*]] PASS=[[N]] FAIL=0 SKIP=0

module {
  handshake.func @top(%arg0: none, ...) -> (none) attributes {argNames = ["inCtrl"], resNames = ["outCtrl"]}{
    %3 = buffer [1] seq %f#1 {initValues = [0]}: none
    %ctrlOut = join %3, %arg0 : none, none
    %f:2 = fork [2] %ctrlOut : none
    return %f#0 : none
  }
}

