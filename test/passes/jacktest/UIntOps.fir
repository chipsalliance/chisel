; RUN: firrtl -i %s -o %s.flo -X flo -p cTwd | tee %s.out | FileCheck %s
; CHECK: Done!

circuit UIntOps : 
  module UIntOps : 
    input b : UInt<16>
    input a : UInt<16>
    output addout : UInt<16>
    output subout : UInt<16>
    output timesout : UInt<16>
    output divout : UInt<16>
    output modout : UInt<16>
    output lshiftout : UInt<16>
    output rshiftout : UInt<16>
    output lessout : UInt<1>
    output greatout : UInt<1>
    output eqout : UInt<1>
    output noteqout : UInt<1>
    output lesseqout : UInt<1>
    output greateqout : UInt<1>
    
    node T_31 = add-wrap(a, b)
    addout := T_31
    node T_32 = sub-wrap(a, b)
    subout := T_32
    node T_33 = mul(a, b)
    timesout := T_33
    node T_34 = div(a, b)
    divout := T_34
    node T_35 = div(a, b)
    modout := T_35
    node T_36 = shl(a, 12)
    lshiftout := T_36
    node T_37 = shr(a, 8)
    rshiftout := T_37
    node T_38 = lt(a, b)
    lessout := T_38
    node T_39 = gt(a, b)
    greatout := T_39
    node T_40 = eq(a, b)
    eqout := T_40
    node T_41 = neq(a, b)
    noteqout := T_41
    node T_42 = leq(a, b)
    lesseqout := T_42
    node T_43 = geq(a, b)
    greateqout := T_43
