; RUN: firrtl -i %s -o %s.flo -X flo -p c 2>&1 | tee %s.out | FileCheck %s
;CHECK: Done!
circuit Mul : 
  module Mul : 
    input x : UInt<2>
    input y : UInt<2>
    output z : UInt<4>
    
    wire tbl : UInt<4>[16]
    tbl[0] := UInt<4>(0)
    tbl[1] := UInt<4>(0)
    tbl[2] := UInt<4>(0)
    tbl[3] := UInt<4>(0)
    tbl[4] := UInt<4>(0)
    tbl[5] := UInt<4>(1)
    tbl[6] := UInt<4>(2)
    tbl[7] := UInt<4>(3)
    tbl[8] := UInt<4>(0)
    tbl[9] := UInt<4>(2)
    tbl[10] := UInt<4>(4)
    tbl[11] := UInt<4>(6)
    tbl[12] := UInt<4>(0)
    tbl[13] := UInt<4>(3)
    tbl[14] := UInt<4>(6)
    tbl[15] := UInt<4>(9)
    node T_42 = shl(x, 2)
    node T_43 = or(T_42, y)
    infer accessor T_44 = tbl[T_43]
    z := T_44
