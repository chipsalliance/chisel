; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
circuit Top : 
  module Top : 
    input clk : Clock
    input reset : UInt<1>
    input a : { w : UInt<42>, x : UInt<30>}[2]
    input b : { w : UInt<42>, x : UInt<30>}[2]
    input p: UInt<1>
    input q: UInt<1>
    output c : { w : UInt<42>, x : UInt<30>}[2]
    output d : { w : UInt<42>, x : UInt<30>}[2]
    
    c is invalid
    when p :
       when q :
          c <= a
       else :
          c <= b
    d <= validif(p,b)

;CHECK: Done!
