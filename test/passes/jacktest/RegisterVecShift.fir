; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
;CHECK: Done!
circuit RegisterVecShift : 
  module RegisterVecShift : 
    input load : UInt<1>
    input clk : Clock
    input reset : UInt<1>
    output out : UInt<4>
    input shift : UInt<1>
    input ins : UInt<4>[4]
    
    reg delays : UInt<4>[4],clk,reset,delays
    when reset : 
      wire T_33 : UInt<4>[4]
      T_33[0] <= UInt<4>(0)
      T_33[1] <= UInt<4>(0)
      T_33[2] <= UInt<4>(0)
      T_33[3] <= UInt<4>(0)
      delays <= T_33
    when load : 
      delays[0] <= ins[0]
      delays[1] <= ins[1]
      delays[2] <= ins[2]
      delays[3] <= ins[3]
    else : when shift : 
        delays[0] <= ins[0]
        delays[1] <= delays[0]
        delays[2] <= delays[1]
        delays[3] <= delays[2]
    out <= delays[3]
