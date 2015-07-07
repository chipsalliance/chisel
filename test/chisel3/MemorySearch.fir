; RUN: firrtl -i %s -o %s.v -X verilog -p c | tee %s.out | FileCheck %s
;CHECK: Done!

circuit MemorySearch : 
  module MemorySearch : 
    input target : UInt<4>
    output address : UInt<3>
    input en : UInt<1>
    output done : UInt<1>
    
    reg index : UInt<3>
    on-reset index := UInt<3>(0)
    wire elts : UInt<4>[7]
    elts[0] := UInt<4>(0)
    elts[1] := UInt<4>(4)
    elts[2] := UInt<4>(15)
    elts[3] := UInt<4>(14)
    elts[4] := UInt<4>(2)
    elts[5] := UInt<4>(5)
    elts[6] := UInt<4>(13)
    infer accessor elt = elts[index]
    node T_35 = bit-not(en)
    node T_36 = eq(elt, target)
    node T_37 = eq(index, UInt<3>(7))
    node T_38 = bit-or(T_36, T_37)
    node end = bit-and(T_35, T_38)
    when en : index := UInt<1>(0)
    else : 
      node T_39 = bit-not(end)
      when T_39 : 
        node T_40 = add-wrap(index, UInt<1>(1))
        index := T_40
    done := end
    address := index
