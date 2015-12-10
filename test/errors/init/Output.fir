; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
; CHECK: Reference y is not fully initialized.

circuit Top : 
  module Top : 
    input clk : Clock
    wire y : UInt<1>

    when UInt(0) : 
      y <= UInt(1)
