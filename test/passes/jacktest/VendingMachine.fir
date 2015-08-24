; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
;CHECK: Done!
circuit VendingMachine : 
  module VendingMachine : 
    output valid : UInt<1>
    input nickel : UInt<1>
    input dime : UInt<1>
    input clk : Clock
    input reset : UInt<1>
    
    reg state : UInt<3>,clk,reset
    onreset state := UInt<3>(0)
    node T_22 = eq(state, UInt<3>(0))
    when T_22 : 
      when nickel : state := UInt<3>(1)
      when dime : state := UInt<3>(2)
    node T_23 = eq(state, UInt<3>(1))
    when T_23 : 
      when nickel : state := UInt<3>(2)
      when dime : state := UInt<3>(3)
    node T_24 = eq(state, UInt<3>(2))
    when T_24 : 
      when nickel : state := UInt<3>(3)
      when dime : state := UInt<3>(4)
    node T_25 = eq(state, UInt<3>(3))
    when T_25 : 
      when nickel : state := UInt<3>(4)
      when dime : state := UInt<3>(4)
    node T_26 = eq(state, UInt<3>(4))
    when T_26 : state := UInt<3>(0)
    node T_27 = eq(state, UInt<3>(4))
    valid := T_27
