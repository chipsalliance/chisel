; RUN: firrtl -i %s -o %s.flo -X flo -p c | tee %s.out | FileCheck %s
;CHECK: Done!
circuit Tbl : 
  module Tbl : 
    input i : UInt<16>
    input d : UInt<16>
    input clk : Clock
    output o : UInt<16>
    input we : UInt<1>
    
    cmem m : UInt<10>[256],clk
    o := UInt<1>(0)
    when we : 
      infer accessor T_13 = m[i]
      node T_14 = bits(d, 9, 0)
      T_13 := T_14
    else : 
      infer accessor T_15 = m[i]
      o := T_15
