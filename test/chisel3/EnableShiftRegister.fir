;RUN: firrtl %s abcefghipjklmno c | tee %s.out | FileCheck %s
;CHECK: To Flo
circuit EnableShiftRegister : 
  module EnableShiftRegister : 
    input in : UInt(4)
    output out : UInt(4)
    input shift : UInt(1)
    
    node T_14 = UInt(0, 4)
    reg r0 : UInt(4)
    r0.init := T_14
    node T_15 = UInt(0, 4)
    reg r1 : UInt(4)
    r1.init := T_15
    node T_16 = UInt(0, 4)
    reg r2 : UInt(4)
    r2.init := T_16
    node T_17 = UInt(0, 4)
    reg r3 : UInt(4)
    r3.init := T_17
    when shift : 
      r0 := in
      r1 := r0
      r2 := r1
      r3 := r2
    out := r3
;CHECK: Finished To Flo
