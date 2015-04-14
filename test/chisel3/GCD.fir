circuit GCD : 
  module GCD : 
    input b : UInt(16)
    input a : UInt(16)
    input e : UInt(1)
    output z : UInt(16)
    output v : UInt(1)
    
    reg x : UInt(16)
    reg y : UInt(16)
    node T_17 : UInt(1) = greater(x, y)
    when T_17 : 
      node T_18 : UInt = sub-mod(x, y)
      x := T_18
     else : 
      node T_19 : UInt = sub-mod(y, x)
      y := T_19
    when e : 
      x := a
      y := b
    z := x
    node T_20 : UInt(1) = UInt(0, 1)
    node T_21 : UInt(1) = equal(y, T_20)
    v := T_21