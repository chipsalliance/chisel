; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
;CHECK: Done!
circuit Tbl : 
  module Tbl : 
    input i : UInt<16>
    input d : UInt<16>
    input clk : Clock
    output o : UInt<16>
    input we : UInt<1>
    
    cmem m : UInt<10>[256]
    o <= UInt<1>(0)
    when we : 
      write mport T_13 = m[i],clk
      node T_14 = bits(d, 9, 0)
      T_13 <= T_14
    else : 
      read mport T_15 = m[i],clk
      o <= T_15
