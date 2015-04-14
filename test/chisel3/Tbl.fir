circuit Tbl : 
  module Tbl : 
    output o : UInt(16)
    input i : UInt(16)
    input d : UInt(16)
    input we : UInt(1)
    
    mem m : UInt(10)[256]
    node T_13 : UInt(1) = UInt(0, 1)
    o := T_13
    when we : 
      accessor T_14 = m[i]
      T_14 := d
     else : 
      accessor T_15 = m[i]
      o := T_15