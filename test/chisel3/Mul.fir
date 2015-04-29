; RUN: firrtl -i %s -o %s.flo -x X -p c | tee %s.out | FileCheck %s
; CHECK: Done!
circuit Mul : 
  module Mul : 
    input y : UInt<2>
    input x : UInt<2>
    output z : UInt<4>
    
    node T_43 = UInt<4>(0)
    node T_44 = UInt<4>(0)
    node T_45 = UInt<4>(0)
    node T_46 = UInt<4>(0)
    node T_47 = UInt<4>(0)
    node T_48 = UInt<4>(1)
    node T_49 = UInt<4>(2)
    node T_50 = UInt<4>(3)
    node T_51 = UInt<4>(0)
    node T_52 = UInt<4>(2)
    node T_53 = UInt<4>(4)
    node T_54 = UInt<4>(6)
    node T_55 = UInt<4>(0)
    node T_56 = UInt<4>(3)
    node T_57 = UInt<4>(6)
    node T_58 = UInt<4>(9)
    wire tbl : UInt<4>[16]
    tbl[0] := T_43
    tbl[1] := T_44
    tbl[2] := T_45
    tbl[3] := T_46
    tbl[4] := T_47
    tbl[5] := T_48
    tbl[6] := T_49
    tbl[7] := T_50
    tbl[8] := T_51
    tbl[9] := T_52
    tbl[10] := T_53
    tbl[11] := T_54
    tbl[12] := T_55
    tbl[13] := T_56
    tbl[14] := T_57
    tbl[15] := T_58
    node T_60 = shl(x, 2)
    node T_61 = bit-or(T_60, y)
    accessor T_62 = tbl[T_61]
    z := T_62
