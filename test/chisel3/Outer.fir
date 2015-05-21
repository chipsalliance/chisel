; RUN: firrtl -i %s -o %s.flo -X flo -p c | tee %s.out | FileCheck %s
;CHECK: Done!

circuit Outer : 
  module Inner : 
    input in : UInt<8>
    output out : UInt<8>
    
    node T_15 = add-wrap(in, UInt<1>(1))
    out := T_15
  module Outer : 
    input in : UInt<8>
    output out : UInt<8>
    
    inst T_16 of Inner
    T_16.in := in
    node T_17 = mul(T_16.out, UInt<2>(2))
    node T_18 = bits(T_17, 7, 0)
    out := T_18
