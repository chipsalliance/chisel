; RUN: firrtl -i %s -o %s.flo -x X -p c | tee %s.out | FileCheck %s
; CHECK: Done!
circuit Outer : 
  module Inner : 
    input in : UInt(8)
    output out : UInt(8)
    
    node T_14 = UInt(1, 1)
    node T_15 = add(in, T_14)
    out := T_15
  module Outer : 
    input in : UInt(8)
    output out : UInt(8)
    
    inst T_16 of Inner
    T_16.in := in
    node T_17 = UInt(2, 2)
    node T_18 = mul(T_16.out, T_17)
    out := T_18
