; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
; CHECK: Done!
circuit Poison :
  module Poison :
    input clk : Clock
    input reset : UInt<1>
    input index : UInt<7>
    input p : UInt<1>
    output out : {x : UInt<10>, y : UInt<10>}
     
    poison q : {x : UInt<10>, y : UInt<10>}
    smem m : {x : UInt<10>, y : UInt<10>}[128],clk
    infer accessor r = m[index]
    when p : 
      out := r
    else :
      out := q

