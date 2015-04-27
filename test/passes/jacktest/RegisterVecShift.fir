; RUN: firrtl -i %s -o %s.flo -x X -p cd | tee %s.out | FileCheck %s
; CHECK: Done!

circuit RegisterVecShift : 
  module RegisterVecShift : 
    input load : UInt<1>
    output out : UInt<4>
    input shift : UInt<1>
    input ins : UInt<4>[4]
    
    reg delays : UInt<4>[4]
    when reset : 
      node T_38 = UInt<4>(0)
      node T_39 = UInt<4>(0)
      node T_40 = UInt<4>(0)
      node T_41 = UInt<4>(0)
      wire T_42 : UInt<4>[4]
      T_42[0] := T_38
      T_42[1] := T_39
      T_42[2] := T_40
      T_42[3] := T_41
      delays := T_42
    node T_43 = UInt<3>(5)
    node T_44 = bit-and(Pad(T_43,?), Pad(load,?))
    node T_45 = UInt<3>(4)
    node T_46 = eq(Pad(T_44,?), Pad(T_45,?))
    when T_46 : 
      delays[0] := Pad(ins[0],?)
      delays[1] := Pad(ins[1],?)
      delays[2] := Pad(ins[2],?)
      delays[3] := Pad(ins[3],?)
    else : when shift : 
        delays[0] := Pad(ins[0],?)
        delays[1] := Pad(delays[0],?)
        delays[2] := Pad(delays[1],?)
        delays[3] := Pad(delays[2],?)
    out := Pad(delays[3],?)
