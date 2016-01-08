; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
;CHECK: Done!
circuit MemorySearch : 
  module MemorySearch : 
    input target : UInt<4>
    output address : UInt<3>
    input en : UInt<1>
    input clk : Clock
    input reset : UInt<1>
    output done : UInt<1>
    
    reg index : UInt<3>,clk,reset,UInt<3>(0)
    wire elts : UInt<4>[7]
    elts[0] <= UInt<4>(0)
    elts[1] <= UInt<4>(4)
    elts[2] <= UInt<4>(15)
    elts[3] <= UInt<4>(14)
    elts[4] <= UInt<4>(2)
    elts[5] <= UInt<4>(5)
    elts[6] <= UInt<4>(13)
    node elt = elts[index]
    node T_35 = not(en)
    node T_36 = eq(elt, target)
    node T_37 = eq(index, UInt<3>(7))
    node T_38 = or(T_36, T_37)
    node end = and(T_35, T_38)
    when en : index <= UInt<1>(0)
    else : 
      node T_39 = not(end)
      when T_39 : 
        node T_40 = addw(index, UInt<1>(1))
        index <= T_40
    done <= end
    address <= index
