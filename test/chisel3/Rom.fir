circuit Rom : 
  module Rom : 
    output out : UInt(8)
    input addr : UInt(8)
    
    node T_24 : UInt(8) = UInt(0, 8)
    node T_25 : UInt(8) = UInt(1, 8)
    node T_26 : UInt(8) = UInt(2, 8)
    node T_27 : UInt(8) = UInt(3, 8)
    node T_28 : UInt(8) = UInt(4, 8)
    node T_29 : UInt(8) = UInt(5, 8)
    node T_30 : UInt(8) = UInt(6, 8)
    node T_31 : UInt(8) = UInt(7, 8)
    wire r : UInt(8)[8]
    r.0 := T_24
    r.1 := T_25
    r.2 := T_26
    r.3 := T_27
    r.4 := T_28
    r.5 := T_29
    r.6 := T_30
    r.7 := T_31
    accessor T_32 = r[addr]
    out := T_32