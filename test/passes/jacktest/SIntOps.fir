; RUN: firrtl -i %s -o %s.flo -X flo -p cTwd | tee %s.out | FileCheck %s
; CHECK: Done!

circuit SIntOps : 
  module SIntOps : 
    input b : SInt<16>
    input a : SInt<16>
    output addout : SInt<16>
    output subout : SInt<16>
    output timesout : SInt<16>
    output divout : SInt<16>
    output modout : SInt<16>
    output lshiftout : SInt<16>
    output rshiftout : SInt<16>
    output lessout : UInt<1>
    output greatout : UInt<1>
    output eqout : UInt<1>
    output noteqout : UInt<1>
    output lesseqout : UInt<1>
    output greateqout : UInt<1>
    output negout : SInt<16>
    
    node T_35 = add-wrap(a, b)
    addout := T_35
    node T_36 = sub-wrap(a, b)
    subout := T_36
    node T_37 = mul(a, b)
    node T_38 = bits(T_37, 15, 0)
    timesout := T_38
    node T_39 = div(a, b)
    divout := T_39
    node T_40 = div(a, b)
    modout := T_40
    node T_41 = shl(a, 12)
    node T_42 = bits(T_41, 15, 0)
    lshiftout := T_42
    node T_43 = shr(a, 8)
    rshiftout := T_43
    node T_44 = lt(a, b)
    lessout := T_44
    node T_45 = gt(a, b)
    greatout := T_45
    node T_46 = eq(a, b)
    eqout := T_46
    node T_47 = neq(a, b)
    noteqout := T_47
    node T_48 = leq(a, b)
    lesseqout := T_48
    node T_49 = geq(a, b)
    greateqout := T_49
    node T_50 = neg(a)
    negout := T_50
