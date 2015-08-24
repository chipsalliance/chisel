; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
;CHECK: Done!
circuit EnableShiftRegister : 
  module EnableShiftRegister : 
    input in : UInt<4>
    input clk : Clock
    input reset : UInt<1>
    output out : UInt<4>
    input shift : UInt<1>
    
    reg r0 : UInt<4>,clk,reset
    onreset r0 := UInt<4>(0)
    reg r1 : UInt<4>,clk,reset
    onreset r1 := UInt<4>(0)
    reg r2 : UInt<4>,clk,reset
    onreset r2 := UInt<4>(0)
    reg r3 : UInt<4>,clk,reset
    onreset r3 := UInt<4>(0)
    when shift : 
      r0 := in
      r1 := r0
      r2 := r1
      r3 := r2
    out := r3
