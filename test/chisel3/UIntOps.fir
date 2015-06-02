; RUN: firrtl -i %s -o %s.v -X verilog -p c | tee %s.out | FileCheck %s
;CHECK: Done!

circuit UIntOps : 
  module UIntOps : 
    input a : UInt<16>
    input b : UInt<16>
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
    
    node T_38 = add-wrap(a, b)
    addout := T_38
    node T_39 = sub-wrap(a, b)
    subout := T_39
    node T_40 = mul(a, b)
    node T_41 = bits(T_40, 15, 0)
    timesout := T_41
    node T_42 = eq(b, UInt<1>(0))
    node T_43 = mux(T_42, UInt<1>(1), b)
    node T_44 = div(a, T_43)
    divout := T_44
    modout := UInt<1>(0)
    node T_45 = bits(b, 3, 0)
    node T_46 = dshl(a, T_45)
    node T_47 = bits(T_46, 15, 0)
    lshiftout := T_47
    node T_48 = dshr(a, b)
    rshiftout := T_48
    node T_49 = lt(a, b)
    lessout := T_49
    node T_50 = gt(a, b)
    greatout := T_50
    node T_51 = eq(a, b)
    eqout := T_51
    node T_52 = neq(a, b)
    noteqout := T_52
    node T_53 = leq(a, b)
    lesseqout := T_53
    node T_54 = geq(a, b)
    greateqout := T_54
