; RUN: firrtl -i %s -o %s.v -X verilog -p c | tee %s.out | FileCheck %s
;CHECK: Done!

circuit SIntOps : 
  module SIntOps : 
    input a : SInt<16>
    input b : SInt<16>
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
    
    node ub = as-UInt(b)
    node T_38 = add-wrap(a, b)
    addout := T_38
    node T_39 = sub-wrap(a, b)
    subout := T_39
    node T_40 = mul(a, b)
    node T_41 = bits(as-UInt(T_40), 15, 0)
    timesout := as-SInt(T_41)
    node T_42 = mul(a, b)
    node T_43 = bits(as-UInt(T_42), 15, 0)
    divout := as-SInt(T_43)
    modout := SInt<1>(0)
    node T_44 = shl(a, 12)
    node T_45 = bits(as-UInt(T_44), 15, 0)
    lshiftout := as-SInt(T_45)
    node T_46 = shr(a, 8)
    rshiftout := T_46
    node T_47 = lt(a, b)
    lessout := T_47
    node T_48 = gt(a, b)
    greatout := T_48
    node T_49 = eq(a, b)
    eqout := T_49
    node T_50 = neq(a, b)
    noteqout := T_50
    node T_51 = leq(a, b)
    lesseqout := T_51
    node T_52 = geq(a, b)
    greateqout := T_52
    node T_53 = sub-wrap(SInt<1>(0), a)
    negout := T_53
