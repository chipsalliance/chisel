circuit UIntOps : 
  module UIntOps : 
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
    
    node T_32 : UInt(16) = add(a, b)
    addout := T_32
    node T_33 : UInt(16) = sub(a, b)
    subout := T_33
    node T_34 : UInt(16) = times(a, b)
    timesout := T_34
    node T_35 : UInt(16) = divide(a, b)
    divout := T_35
    node T_36 : UInt(16) = divide(a, b)
    modout := T_36
    node T_37 : UInt(16) = shift-left(a, 12)
    lshiftout := T_37
    node T_38 : UInt(16) = shift-right(a, 8)
    rshiftout := T_38
    node T_39 : UInt(1) = less(a, b)
    lessout := T_39
    node T_40 : UInt(1) = greater(a, b)
    greatout := T_40
    node T_41 : UInt(1) = equal(a, b)
    eqout := T_41
    node T_42 : UInt(1) = not-equal(a, b)
    noteqout := T_42
    node T_43 : UInt(1) = less-eq(a, b)
    lesseqout := T_43
    node T_44 : UInt(1) = greater-eq(a, b)
    greateqout := T_44