// REQUIRES: verilator
// RUN: circt-opt %s --convert-fsm-to-sv --canonicalize --lower-seq-to-sv --export-split-verilog -o %t2.mlir
// RUN: circt-rtl-sim.py --compileargs="-I%T/.." top.sv %S/driver.cpp --no-default-driver | FileCheck %s
// CHECK: out: A
// CHECK: out: B
// CHECK: out: B
// CHECK: out: C
// CHECK: out: B
// CHECK: out: C
// CHECK: out: A

fsm.machine @top(%arg0: i1, %arg1: i1) -> (i8) attributes {initialState = "A"} {

  fsm.state @A output  {
    %c_0 = hw.constant 0 : i8
    fsm.output %c_0 : i8
  } transitions {
    fsm.transition @B
  }

  fsm.state @B output  {
    %c_1 = hw.constant 1 : i8
    fsm.output %c_1 : i8
  } transitions {
    fsm.transition @C guard {
      %g = comb.and %arg0, %arg1 : i1
      fsm.return %g
    }
  }

  fsm.state @C output  {
    %c_2 = hw.constant 2 : i8
    fsm.output %c_2 : i8
  } transitions {
    fsm.transition @A guard {
      %g = comb.and %arg0, %arg1 : i1
      fsm.return %g
    }
    fsm.transition @B
  }
}
