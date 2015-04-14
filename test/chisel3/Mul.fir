circuit Mul : 
  module Mul : 
    input y : UInt(2)
    input x : UInt(2)
    output z : UInt(4)
    
    node T_44 : UInt(4) = UInt(0, 4)
    node T_45 : UInt(4) = UInt(0, 4)
    node T_46 : UInt(4) = UInt(0, 4)
    node T_47 : UInt(4) = UInt(0, 4)
    node T_48 : UInt(4) = UInt(0, 4)
    node T_49 : UInt(4) = UInt(1, 4)
    node T_50 : UInt(4) = UInt(2, 4)
    node T_51 : UInt(4) = UInt(3, 4)
    node T_52 : UInt(4) = UInt(0, 4)
    node T_53 : UInt(4) = UInt(2, 4)
    node T_54 : UInt(4) = UInt(4, 4)
    node T_55 : UInt(4) = UInt(6, 4)
    node T_56 : UInt(4) = UInt(0, 4)
    node T_57 : UInt(4) = UInt(3, 4)
    node T_58 : UInt(4) = UInt(6, 4)
    node T_59 : UInt(4) = UInt(9, 4)
    wire tbl : UInt(4)[16]
    tbl.0 := T_44
    tbl.1 := T_45
    tbl.2 := T_46
    tbl.3 := T_47
    tbl.4 := T_48
    tbl.5 := T_49
    tbl.6 := T_50
    tbl.7 := T_51
    tbl.8 := T_52
    tbl.9 := T_53
    tbl.10 := T_54
    tbl.11 := T_55
    tbl.12 := T_56
    tbl.13 := T_57
    tbl.14 := T_58
    tbl.15 := T_59
    node T_60 : UInt(2) = UInt(2, 2)
    node T_61 : UInt(2) = shift-left(x, T_60)
    node T_62 : UInt(2) = bit-or(T_61, y)
    accessor T_63 = tbl[T_62]
    z := T_63