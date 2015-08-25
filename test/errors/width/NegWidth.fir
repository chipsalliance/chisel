; RUN: firrtl -i %s -o %s.v -X verilog -p ciwT 2>&1 | tee %s.out | FileCheck %s
; CHECK: Width cannot be negative or zero.

circuit Top : 
  module Top : 
    output y : UInt

    wire x : UInt<2>
    y := shr(x,4)
