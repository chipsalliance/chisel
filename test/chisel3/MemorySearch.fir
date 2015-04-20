; RUN: firrtl -i %s -o %s.flo -x X -p c | tee %s.out | FileCheck %s
; CHECK: Done!
circuit MemorySearch : 
  module MemorySearch : 
    input target : UInt(4)
    output address : UInt(3)
    input en : UInt(1)
    output odone : UInt(1)
    
    node T_35 = UInt(0, 3)
    reg index : UInt(3)
    index.init := T_35
    node T_36 = UInt(0, 1)
    node T_37 = UInt(4, 3)
    node T_38 = UInt(15, 4)
    node T_39 = UInt(14, 4)
    node T_40 = UInt(2, 2)
    node T_41 = UInt(5, 3)
    node T_42 = UInt(13, 4)
    wire elts : UInt(1)[7]
    elts.0 := T_36
    elts.1 := T_37
    elts.2 := T_38
    elts.3 := T_39
    elts.4 := T_40
    elts.5 := T_41
    elts.6 := T_42
    accessor elt = elts[index]
    node T_43 = bit-not(en)
    node T_44 = eq(elt, target)
    node T_45 = UInt(7, 3)
    node T_46 = eq(index, T_45)
    node T_47 = bit-or(T_44, T_46)
    node done = bit-and(T_43, T_47)
    when en : 
      node T_48 = UInt(0, 1)
      index := T_48
    else : 
      node T_49 = bit-not(done)
      when T_49 : 
        node T_50 = UInt(1, 1)
        node T_51 = add(index, T_50)
        index := T_51
    odone := done
    address := index
