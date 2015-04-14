circuit Outer : 
  module Inner : 
    input in : UInt(8)
    output out : UInt(8)
    
    node T_15 : UInt(1) = UInt(1, 1)
    node T_16 : UInt(8) = add(in, T_15)
    out := T_16
  module Outer : 
    input in : UInt(8)
    output out : UInt(8)
    
    instance T_17 of Inner
    T_17.in := in
    node T_18 : UInt(2) = UInt(2, 2)
    node T_19 : UInt(8) = times(T_17.out, T_18)
    out := T_19