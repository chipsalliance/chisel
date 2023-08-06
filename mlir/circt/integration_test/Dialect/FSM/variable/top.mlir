// REQUIRES: verilator
// RUN: circt-opt %s --convert-fsm-to-sv --canonicalize --lower-seq-to-sv --export-split-verilog -o %t2.mlir
// RUN: circt-rtl-sim.py --compileargs="-I%T/.." top.sv %S/driver.cpp --no-default-driver | FileCheck %s

// A simple FSM with an internal variable. The FSM counts up to 5 and then resets
// the counter to 0, always emitting the current counter value on the output.

// CHECK:      out: 0
// CHECK-NEXT: out: 0
// CHECK-NEXT: out: 1
// CHECK-NEXT: out: 2
// CHECK-NEXT: out: 3
// CHECK-NEXT: out: 4
// CHECK-NEXT: out: 5
// CHECK-NEXT: out: 0


fsm.machine @top(%go: i1) -> (i16) attributes {initialState = "A"} {
  %cnt = fsm.variable "cnt" {initValue = 0 : i16} : i16
  %c_0 = hw.constant 0 : i16
  %c_1 = hw.constant 1 : i16
  %c_5 = hw.constant 5 : i16

  fsm.state @A output  {
    fsm.output %cnt : i16
  } transitions {
    fsm.transition @B guard {
      fsm.return %go
    }
  }

  fsm.state @B output  {
    fsm.output %cnt : i16
  } transitions {
    fsm.transition @A guard {
      %eq = comb.icmp eq %cnt, %c_5 : i16
      fsm.return %eq
    } action {
      fsm.update %cnt, %c_0 : i16
    }

    fsm.transition @B action {
      %add1 = comb.add %cnt, %c_1 : i16
      fsm.update %cnt, %add1 : i16
    }
  }
}
