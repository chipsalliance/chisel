; RUN: firrtl -i %s -o %s.flo -x X -p cd | tee %s.out | FileCheck %s
; CHECK: Done!

circuit MemorySearch : 
  module MemorySearch : 
    input target : UInt<4>
    output address : UInt<3>
    input en : UInt<1>
    output done : UInt<1>
    
    node T_35 = UInt<3>(0)
    reg index : UInt<3>
    on-reset index := T_35
    node T_36 = UInt<1>(0)
    node T_37 = UInt<3>(4)
    node T_38 = UInt<4>(15)
    node T_39 = UInt<4>(14)
    node T_40 = UInt<2>(2)
    node T_41 = UInt<3>(5)
    node T_42 = UInt<4>(13)
    wire elts : UInt<1>[7]
    elts[0] := Pad(T_36,?)
    elts[1] := Pad(T_37,?)
    elts[2] := Pad(T_38,?)
    elts[3] := Pad(T_39,?)
    elts[4] := Pad(T_40,?)
    elts[5] := Pad(T_41,?)
    elts[6] := Pad(T_42,?)
    accessor elt = elts[index]
    node T_43 = bit-not(en)
    node T_44 = eq(Pad(elt,?), Pad(target,?))
    node T_45 = UInt<3>(7)
    node T_46 = eq(Pad(index,?), Pad(T_45,?))
    node T_47 = bit-or(T_44, T_46)
    node end = bit-and(T_43, T_47)
    when en : 
      node T_48 = UInt<1>(0)
      index := Pad(T_48,?)
    else : 
      node T_49 = bit-not(end)
      when T_49 : 
        node T_50 = UInt<1>(1)
        node T_51 = add-wrap(Pad(index,?), Pad(T_50,?))
        index := Pad(T_51,?)
    done := Pad(end,?)
    address := Pad(index,?)
