; RUN: firrtl -i %s -o %s.v -X verilog -p c | tee %s.out | FileCheck %s
; CHECK: Done!

circuit Test : 
  module Test : 
     input clk : Clock
     input reset : UInt<1>
     input falling : UInt<1>

     reg hold : UInt<100>, clk, UInt(1)

     hold := UInt("h42")
     when reset :
        hold := UInt("h7f")
     else :
        when falling :
           hold := UInt("h8f")

