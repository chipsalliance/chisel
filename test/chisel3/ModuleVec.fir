circuit ModuleVec : 
  module PlusOne : 
    input in : UInt(32)
    output out : UInt(32)
    
    node T_34 : UInt(1) = UInt(1, 1)
    node T_35 : UInt(32) = add(in, T_34)
    out := T_35
  module PlusOne_26 : 
    input in : UInt(32)
    output out : UInt(32)
    
    node T_36 : UInt(1) = UInt(1, 1)
    node T_37 : UInt(32) = add(in, T_36)
    out := T_37
  module ModuleVec : 
    input ins : UInt(32)[2]
    output outs : UInt(32)[2]
    
    instance T_38 of PlusOne
    instance T_39 of PlusOne_26
    pluses.0.in := ins.0
    outs.0 := pluses.0.out
    pluses.1.in := ins.1
    outs.1 := pluses.1.out