; RUN: firrtl -i %s -o %s.flo -x X -p c | tee %s.out | FileCheck %s
; CHECK: Done!
circuit RegisterVecShift : 
  module RegisterVecShift : 
    input load : UInt(1)
    output out : UInt(4)
    input shift : UInt(1)
    input ins : UInt(4)[4]
    
    reg delays : UInt(4)[4]
    when reset : 
      node T_39 = UInt(0, 4)
      node T_40 = UInt(0, 4)
      node T_41 = UInt(0, 4)
      node T_42 = UInt(0, 4)
      wire T_43 : UInt(4)[4]
      T_43.0 := T_39
      T_43.1 := T_40
      T_43.2 := T_41
      T_43.3 := T_42
      delays := T_43
    node T_44 = UInt(5, 3)
    node T_45 = bit-and(T_44, load)
    node T_46 = UInt(4, 3)
    node T_47 = eq(T_45, T_46)
    when T_47 : 
      delays.0 := ins.0
      delays.1 := ins.1
      delays.2 := ins.2
      delays.3 := ins.3
     else : when shift : 
        delays.0 := ins.0
        delays.1 := delays.0
        delays.2 := delays.1
        delays.3 := delays.2
    out := delays.3
