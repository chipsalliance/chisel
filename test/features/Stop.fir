; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s

;CHECK: Expand Whens

circuit Top : 
  module Top : 
    input p : UInt<1>
    input q : UInt<1>
    input clk : Clock
    when p :
       stop(clk,UInt(1),0)
    when q :
       stop(clk,UInt(1),1)
    stop(clk,UInt(1),3)

;CHECK: stop(clk, p, 0)
;CHECK: stop(clk, q, 1)
;CHECK: stop(clk, UInt<1>("h1"), 3)   
;CHECK: Done!

