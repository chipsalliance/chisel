circuit MemorySearch : 
  module MemorySearch : 
    input target : UInt(4)
    output address : UInt(3)
    input en : UInt(1)
    output done : UInt(1)
    
    node T_36 : UInt(3) = UInt(0, 3)
    reg index : UInt(3)
    index.init := T_36
    node T_37 : UInt(1) = UInt(0, 1)
    node T_38 : UInt(3) = UInt(4, 3)
    node T_39 : UInt(4) = UInt(15, 4)
    node T_40 : UInt(4) = UInt(14, 4)
    node T_41 : UInt(2) = UInt(2, 2)
    node T_42 : UInt(3) = UInt(5, 3)
    node T_43 : UInt(4) = UInt(13, 4)
    wire elts : UInt(1)[7]
    elts.0 := T_37
    elts.1 := T_38
    elts.2 := T_39
    elts.3 := T_40
    elts.4 := T_41
    elts.5 := T_42
    elts.6 := T_43
    accessor elt = elts[index]
    node T_44 : UInt(1) = bit-not(en)
    node T_45 : UInt(1) = equal(elt, target)
    node T_46 : UInt(3) = UInt(7, 3)
    node T_47 : UInt(1) = equal(index, T_46)
    node T_48 : UInt(1) = bit-or(T_45, T_47)
    node done : UInt(1) = bit-and(T_44, T_48)
    when en : 
      node T_49 : UInt(1) = UInt(0, 1)
      index := T_49
     else : 
      node T_50 : UInt(1) = bit-not(done)
      when T_50 : 
        node T_51 : UInt(1) = UInt(1, 1)
        node T_52 : UInt(3) = add(index, T_51)
        index := T_52
    done := done
    address := index
