;RUN: firrtl %s abcefghipjklmno c | tee %s.out | FileCheck %s
;CHECK: To Flo
circuit ComplexAssign : 
  module ComplexAssign : 
    input in : {re : UInt(10), im : UInt(10)}
    output out : {re : UInt(10), im : UInt(10)}
    input e : UInt(1)
    when e : 
      wire T_19 : {re : UInt(10), im : UInt(10)}
      T_19 := in
      out.re := T_19.re
      out.im := T_19.im
     else : 
      node T_20 = UInt(0, 1)
      out.re := T_20
      node T_21 = UInt(0, 1)
      out.im := T_21
;CHECK: Finished To Flo
