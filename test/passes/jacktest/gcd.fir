; RUN: firrtl -i %s -o %s.flo -X flo -p c 2>&1 | tee %s.out | FileCheck %s
;CHECK: Done!
circuit GCD : 
  module GCD : 
    input e : UInt<1>
    input clk : Clock
    input reset : UInt<1>
    output z : UInt<16>
    output v : UInt<1>
    input a : UInt<16>
    input b : UInt<16>
    
    reg x : UInt<16>,clk,reset
    reg y : UInt<16>,clk,reset
    node T_17 = gt(x, y)
    when T_17 : 
      node T_18 = subw(x, y)
      x := T_18
    else : 
      node T_19 = subw(y, x)
      y := T_19
    when e : 
      x := a
      y := b
    z := x
    node T_20 = eq(y, UInt<1>(0))
    v := T_20
