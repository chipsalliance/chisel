; RUN: firrtl -i %s -o %s.flo -x X -p cTwd | tee %s.out | FileCheck %s
; CHECK: Done!
circuit Tbl : 
  module Tbl : 
    output o : UInt(16)
    input i : UInt(16)
    input d : UInt(16)
    input we : UInt(1)
    
    mem m : UInt(10)[256]
    node T_12 = UInt(0, 1)
    o := Pad(T_12,?)
    when we : 
      accessor T_13 = m[i]
      T_13 := Pad(d,?)
    else : 
      accessor T_14 = m[i]
      o := Pad(T_14,?)
