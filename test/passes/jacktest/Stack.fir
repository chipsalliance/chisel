; RUN: firrtl -i %s -o %s.flo -X flo -p c 2>&1 | tee %s.out | FileCheck %s
;CHECK: Done!
circuit Stack : 
  module Stack : 
    input push : UInt<1>
    input pop : UInt<1>
    input en : UInt<1>
    input clk : Clock
    input reset : UInt<1>
    output dataOut : UInt<32>
    input dataIn : UInt<32>
    
    cmem stack_mem : UInt<32>[16],clk
    reg sp : UInt<5>,clk,reset
    onreset sp := UInt<5>(0)
    reg out : UInt<32>,clk,reset
    onreset out := UInt<32>(0)
    when en : 
      node T_30 = lt(sp, UInt<5>(16))
      node T_31 = and(push, T_30)
      when T_31 : 
        infer accessor T_32 = stack_mem[sp]
        T_32 := dataIn
        node T_33 = addw(sp, UInt<1>(1))
        sp := T_33
      else : 
        node T_34 = gt(sp, UInt<1>(0))
        node T_35 = and(pop, T_34)
        when T_35 : 
          node T_36 = subw(sp, UInt<1>(1))
          sp := T_36
      node T_37 = gt(sp, UInt<1>(0))
      when T_37 : 
        node T_38 = subw(sp, UInt<1>(1))
        infer accessor T_39 = stack_mem[T_38]
        out := T_39
    dataOut := out
