; RUN: firrtl -i %s -o %s.flo -x X -p c | tee %s.out | FileCheck %s
; CHECK: Done!
circuit ModuleVec : 
  module PlusOne : 
    input in : UInt<32>
    output out : UInt<32>
    
    node T_33 = UInt<1>(1)
    node T_34 = add-wrap(Pad(in,?), Pad(T_33,?))
    out := Pad(T_34,?)
  module PlusOne_25 : 
    input in : UInt<32>
    output out : UInt<32>
    
    node T_35 = UInt<1>(1)
    node T_36 = add-wrap(Pad(in,?), Pad(T_35,?))
    out := Pad(T_36,?)
  module ModuleVec : 
    input ins : UInt<32>[2]
    output outs : UInt<32>[2]
    
    inst T_37 of PlusOne
    inst T_38 of PlusOne_25
    wire pluses : { in : UInt<32>, flip out : UInt<32>}[2]
    pluses[0] := Pad(T_37,?)
    pluses[1] := Pad(T_38,?)
    pluses[0].in := Pad(ins[0],?)
    outs[0] := Pad(pluses[0].out,?)
    pluses[1].in := Pad(ins[1],?)
    outs[1] := Pad(pluses[1].out,?)
