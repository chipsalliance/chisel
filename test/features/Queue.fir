; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
; CHECK: Done!
circuit Queue :
  module Queue :
    input in : UInt<10>
    output out : UInt<10>
    input clk : Clock
    input reset : UInt<1>
  
    reg r : UInt<10>,clk,reset
    r := in
    out := r
