; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
;CHECK: Done!
circuit ModuleVec : 
  module PlusOne : 
    input in : UInt<32>
    output out : UInt<32>
    
    node T_33 = addw(in, UInt<1>(1))
    out <= T_33
  module PlusOne_25 : 
    input in : UInt<32>
    output out : UInt<32>
    
    node T_34 = addw(in, UInt<1>(1))
    out <= T_34
  module ModuleVec : 
    input ins : UInt<32>[2]
    output outs : UInt<32>[2]
    
    inst T_35 of PlusOne
    inst T_36 of PlusOne_25
    wire pluses : {flip in : UInt<32>, out : UInt<32>}[2]
    pluses[0] <= T_35
    pluses[1] <= T_36
    pluses[0].in <= ins[0]
    outs[0] <= pluses[0].out
    pluses[1].in <= ins[1]
    outs[1] <= pluses[1].out
