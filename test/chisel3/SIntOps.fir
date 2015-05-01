; RUN: firrtl -i %s -o %s.flo -x X -p c | tee %s.out | FileCheck %s
; CHECK: Done!

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
    
    wire ub : UInt
    ub := b
    node T_44 = add-wrap(Pad(a,?), Pad(b,?))
    addout := Pad(T_44,?)
    node T_45 = sub-wrap(Pad(a,?), Pad(b,?))
    subout := Pad(T_45,?)
    node T_46 = mul(Pad(a,?), Pad(b,?))
    node T_47 = bits(T_46, 15, 0)
    timesout := Pad(T_47,?)
    node T_48 = eq(Pad(b,?), Pad(SInt<1>(0),?))
    node T_49 = mux(Pad(T_48,?), Pad(SInt<2>(1),?), Pad(b,?))
    node T_50 = div(Pad(a,?), Pad(T_49,?))
    divout := Pad(T_50,?)
    modout := Pad(UInt<1>(0),?)
    node T_51 = bits(ub, 3, 0)
    node T_52 = dshl(a, T_51)
    node T_53 = bits(T_52, 15, 0)
    wire T_54 : SInt
    T_54 := T_53
    lshiftout := Pad(T_54,?)
    node T_55 = dshr(a, ub)
    wire T_56 : SInt
    T_56 := T_55
    rshiftout := Pad(T_56,?)
    node T_57 = lt(Pad(a,?), Pad(b,?))
    lessout := Pad(T_57,?)
    node T_58 = gt(Pad(a,?), Pad(b,?))
    greatout := Pad(T_58,?)
    node T_59 = eq(Pad(a,?), Pad(b,?))
    eqout := Pad(T_59,?)
    node T_60 = neq(Pad(a,?), Pad(b,?))
    noteqout := Pad(T_60,?)
    node T_61 = leq(Pad(a,?), Pad(b,?))
    lesseqout := Pad(T_61,?)
    node T_62 = geq(Pad(a,?), Pad(b,?))
    greateqout := Pad(T_62,?)
    node T_63 = sub-wrap(Pad(SInt<1>(0),?), Pad(a,?))
    negout := Pad(T_63,?)
