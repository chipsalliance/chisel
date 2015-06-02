; RUN: firrtl -i %s -o %s.flo -X flo -p c | tee %s.out | FileCheck %s
;CHECK: Done!

circuit EnableShiftRegister : 
  module EnableShiftRegister : 
    input in : UInt<4>
    output out : UInt<4>
    input shift : UInt<1>
    
    reg r0 : UInt<4>
    on-reset r0 := UInt<4>(0)
    reg r1 : UInt<4>
    on-reset r1 := UInt<4>(0)
    reg r2 : UInt<4>
    on-reset r2 := UInt<4>(0)
    reg r3 : UInt<4>
    on-reset r3 := UInt<4>(0)
    when shift : 
      r0 := in
      r1 := r0
      r2 := r1
      r3 := r2
    out := r3
