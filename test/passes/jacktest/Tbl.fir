; RUN: firrtl -i %s -o %s.flo -X flo -p c | tee %s.out | FileCheck %s
;CHECK: Done!
circuit Tbl : 
  module Tbl : 
    input i : UInt<16>
    input d : UInt<16>
    output o : UInt<16>
    input we : UInt<1>
    
    mem m : UInt<10>[256]
    o := UInt<1>(0)
    when we : 
      accessor T_13 = m[i]
      node T_14 = bits(d, 9, 0)
      T_13 := T_14
    else : 
      accessor T_15 = m[i]
      o := T_15
