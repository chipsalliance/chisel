circuit Stack : 
  module Stack : 
    input push : UInt(1)
    input pop : UInt(1)
    input en : UInt(1)
    output dataOut : UInt(32)
    input dataIn : UInt(32)
    
    mem stack_mem : UInt(32)[16]
    node T_30 : UInt(5) = UInt(0, 5)
    reg sp : UInt(5)
    sp.init := T_30
    node T_31 : UInt(32) = UInt(0, 32)
    reg out : UInt(32)
    out.init := T_31
    when en : 
      node T_32 : UInt(5) = UInt(16, 5)
      node T_33 : UInt(1) = less(sp, T_32)
      node T_34 : UInt(1) = bit-and(push, T_33)
      when T_34 : 
        accessor T_35 = stack_mem[sp]
        T_35 := dataIn
        node T_36 : UInt(1) = UInt(1, 1)
        node T_37 : UInt = add-mod(sp, T_36)
        sp := T_37
       else : 
        node T_38 : UInt(1) = UInt(0, 1)
        node T_39 : UInt(1) = greater(sp, T_38)
        node T_40 : UInt(1) = bit-and(pop, T_39)
        when T_40 : 
          node T_41 : UInt(1) = UInt(1, 1)
          node T_42 : UInt = sub-mod(sp, T_41)
          sp := T_42
      node T_43 : UInt(1) = UInt(0, 1)
      node T_44 : UInt(1) = greater(sp, T_43)
      when T_44 : 
        node T_45 : UInt(1) = UInt(1, 1)
        node T_46 : UInt = sub-mod(sp, T_45)
        accessor T_47 = stack_mem[T_46]
        out := T_47
    dataOut := out