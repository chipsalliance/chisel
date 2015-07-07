; RUN: firrtl -i %s -o %s.v -X verilog -p c | tee %s.out | FileCheck %s
;CHECK: Done!

circuit Rom : 
  module Rom : 
    output out : UInt<5>
    input addr : UInt<4>
    
    wire r : UInt<5>[16]
    r[0] := UInt<5>(0)
    r[1] := UInt<5>(2)
    r[2] := UInt<5>(4)
    r[3] := UInt<5>(6)
    r[4] := UInt<5>(8)
    r[5] := UInt<5>(10)
    r[6] := UInt<5>(12)
    r[7] := UInt<5>(14)
    r[8] := UInt<5>(16)
    r[9] := UInt<5>(18)
    r[10] := UInt<5>(20)
    r[11] := UInt<5>(22)
    r[12] := UInt<5>(24)
    r[13] := UInt<5>(26)
    r[14] := UInt<5>(28)
    r[15] := UInt<5>(30)
    infer accessor T_39 = r[addr]
    out := T_39
