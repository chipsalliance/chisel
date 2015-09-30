; RUN: firrtl -i %s -m %S/Queue.fir -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
;CHECK: Lower To Ground
circuit Top : 
  module Top : 
    input clk : Clock
    input reset : UInt<1>
    output out : UInt<10>
    
    inst q of Queue
    q.clk := clk
    q.reset := reset
    q.in := UInt(1)
    out := q.out

