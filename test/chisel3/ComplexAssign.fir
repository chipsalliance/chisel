circuit ComplexAssign : 
  module ComplexAssign : 
    input in : {output re : UInt(10), output im : UInt(10)}
    output out : {output re : UInt(10), output im : UInt(10)}
    input e : UInt(1)
    when e : 
      wire T_19 : {output re : UInt(10), output im : UInt(10)}
      T_19 := in
      out.re := T_19.re
      out.im := T_19.im
     else : 
      node T_20 : UInt(1) = UInt(0, 1)
      out.re := T_20
      node T_21 : UInt(1) = UInt(0, 1)
      out.im := T_21