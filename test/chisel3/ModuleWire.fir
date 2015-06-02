; RUN: firrtl -i %s -o %s.v -X verilog -p c | tee %s.out | FileCheck %s
;CHECK: Done!

circuit ModuleWire : 
  module Inc : 
    input in : UInt<32>
    output out : UInt<32>
    
    node T_12 = add-wrap(in, UInt<1>(1))
    out := T_12
  module ModuleWire : 
    input in : UInt<32>
    output out : UInt<32>
    
    inst T_13 of Inc
    T_13.in := in
    out := T_13.out
