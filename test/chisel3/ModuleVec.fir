; RUN: firrtl -i %s -o %s.flo -x X -p c | tee %s.out | FileCheck %s
; CHECK: Done!
circuit ModuleVec : 
  module PlusOne : 
    input in : UInt(32)
    output out : UInt(32)
    
    node T_33 = UInt(1, 1)
    node T_34 = add(in, T_33)
    out := T_34
  module PlusOne_25 : 
    input in : UInt(32)
    output out : UInt(32)
    
    node T_35 = UInt(1, 1)
    node T_36 = add(in, T_35)
    out := T_36
  module ModuleVec : 
    output ins : UInt(32)[2]
    output outs : UInt(32)[2]
    
    inst T_37 of PlusOne
    inst T_38 of PlusOne_25
    wire pluses : {flip in : UInt(32), out : UInt(32)}[2]
    pluses.0 := T_37
    pluses.1 := T_38
    pluses.s.in := ins.s
    outs.0 := pluses.s.out
    pluses.s.in := ins.1
    outs.1 := pluses.1.out
