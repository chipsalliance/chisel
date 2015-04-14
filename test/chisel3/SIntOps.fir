circuit SIntOps : 
  module SIntOps : 
    input b : UInt(16)
    input a : UInt(16)
    output addout : UInt(16)
    output subout : UInt(16)
    output timesout : UInt(16)
    output divout : UInt(16)
    output modout : UInt(16)
    output lshiftout : UInt(16)
    output rshiftout : UInt(16)
    output lessout : UInt(1)
    output greatout : UInt(1)
    output eqout : UInt(1)
    output noteqout : UInt(1)
    output lesseqout : UInt(1)
    output greateqout : UInt(1)
    output negout : UInt(16)
    
    node T_35 : UInt = add-mod(a, b)
    addout := T_35
    node T_36 : UInt = sub-mod(a, b)
    subout := T_36
    node T_37 : UInt = times(a, b)
    node T_38 : UInt = bits(T_37, 15, 0)
    timesout := T_38
    node T_39 : UInt = divide(a, b)
    divout := T_39
    node T_40 : UInt = divide(a, b)
    modout := T_40
    node T_41 : UInt = shift-left(a, 12)
    node T_42 : UInt = bits(T_41, 15, 0)
    lshiftout := T_42
    node T_43 : UInt = shift-right(a, 8)
    rshiftout := T_43
    node T_44 : UInt(1) = less(a, b)
    lessout := T_44
    node T_45 : UInt(1) = greater(a, b)
    greatout := T_45
    node T_46 : UInt(1) = equal(a, b)
    eqout := T_46
    node T_47 : UInt(1) = not-equal(a, b)
    noteqout := T_47
    node T_48 : UInt(1) = less-eq(a, b)
    lesseqout := T_48
    node T_49 : UInt(1) = greater-eq(a, b)
    greateqout := T_49
    node T_50 : UInt = neg(a)
    negout := T_50