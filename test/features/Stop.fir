; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s

;CHECK: Lower To Ground

circuit Top : 
  module Top : 
    input p : UInt<1>
    input q : UInt<1>
    input clk : Clock
    when p :
       stop(clk,0)
    when q :
       stop(clk,1)
    stop(clk,3)

;CHECK: when p : stop(clk, 0)
;CHECK: when q : stop(clk, 1)
;CHECK: stop(clk, 3)   

;CHECK: Done!

