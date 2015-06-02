; RUN: firrtl -i %s -o %s.v -X verilog -p c | tee %s.out | FileCheck %s
;CHECK: Done!

circuit GCD : 
  module GCD : 
    output v : UInt<1>
    input e : UInt<1>
    output z : UInt<16>
    input a : UInt<16>
    input b : UInt<16>
    
    reg x : UInt<16>
    reg y : UInt<16>
    node T_17 = gt(x, y)
    when T_17 : 
      node T_18 = sub-wrap(x, y)
      x := T_18
    else : 
      node T_19 = sub-wrap(y, x)
      y := T_19
    when e : 
      x := a
      y := b
    z := x
    node T_20 = eq(y, UInt<1>(0))
    v := T_20
