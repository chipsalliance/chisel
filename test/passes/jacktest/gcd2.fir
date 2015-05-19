; RUN: firrtl -i %s -o %s.flo -X flo -p cTwd | tee %s.out | FileCheck %s
;CHECK: To Flo
circuit GCD : 
  module GCD : 
    input b : UInt<16>
    input a : UInt<16>
    input e : UInt<1>
    output z : UInt<16>
    output v : UInt<1>
    
    reg x : UInt<16>
    reg y : UInt<16>
    node T_17 = gt(Pad(x,?), Pad(y,?))
    when T_17 : 
      node T_18 = sub-wrap(Pad(x,?), Pad(y,?))
      x := T_18
    else : 
      node T_19 = sub-wrap(Pad(y,?), Pad(x,?))
      y := T_19
    when e : 
      x := a
      y := b
    z := x
    node T_20 = UInt<1>(0)
    node T_21 = eq(Pad(y,?), Pad(T_20,?))
    v := T_21
