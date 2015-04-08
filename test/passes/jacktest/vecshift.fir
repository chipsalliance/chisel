; RUN: firrtl %s abcefghipj c | tee %s.out | FileCheck %s 

; CHECK: Expand Whens

circuit VecShiftRegister : 
  module VecShiftRegister : 
    input load : UInt(1)
    output out : UInt(4)
    input shift : UInt(1)
    input ins : UInt(4)[4]
    
    reg delays : UInt(4)[4]
    when load : 
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
; CHECK: Finished Expand Whens
