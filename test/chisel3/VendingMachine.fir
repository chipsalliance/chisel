circuit VendingMachine : 
  module VendingMachine : 
    output valid : UInt(1)
    input nickel : UInt(1)
    input dime : UInt(1)
    
    node T_34 : UInt(3) = UInt(5, 3)
    wire c : UInt
    c := T_34
    node T_35 : UInt(3) = UInt(0, 3)
    wire sIdle : UInt
    sIdle := T_35
    node T_36 : UInt(3) = UInt(1, 3)
    wire s5 : UInt
    s5 := T_36
    node T_37 : UInt(3) = UInt(2, 3)
    wire s10 : UInt
    s10 := T_37
    node T_38 : UInt(3) = UInt(3, 3)
    wire s15 : UInt
    s15 := T_38
    node T_39 : UInt(3) = UInt(4, 3)
    wire sOk : UInt
    sOk := T_39
    reg state : UInt
    state.init := sIdle
    node T_40 : UInt(1) = equal(state, sIdle)
    when T_40 : 
      when nickel : state := s5
      when dime : state := s10
    node T_41 : UInt(1) = equal(state, s5)
    when T_41 : 
      when nickel : state := s10
      when dime : state := s15
    node T_42 : UInt(1) = equal(state, s10)
    when T_42 : 
      when nickel : state := s15
      when dime : state := sOk
    node T_43 : UInt(1) = equal(state, s15)
    when T_43 : 
      when nickel : state := sOk
      when dime : state := sOk
    node T_44 : UInt(1) = equal(state, sOk)
    when T_44 : state := sIdle
    node T_45 : UInt(1) = equal(state, sOk)
    valid := T_45