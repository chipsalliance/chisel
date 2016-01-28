; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
circuit Top : 
  module Top : 
    input clk : Clock
    input reset : UInt<1>
    input a : UInt<32>
    input p : UInt<1>
    output b : UInt<32>
    reg r1:UInt<32> clk with : 
      reset => (reset, a)
    when p :
      b <= r1
    else :
      b <= r1


;CHECK: Done!
