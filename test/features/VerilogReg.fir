; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
; CHECK: Done!
circuit Poison :
  module Poison :
    input clk : Clock
    input reset : UInt<1>
    input p1 : UInt<1>
    input p2 : UInt<1>
    input p3 : UInt<1>
    reg r : UInt<32>,clk,reset,r
    when p1 :
       r <= UInt(1)
    when p2 :
       r <= UInt(2)
    when p3 :
       r <= UInt(3)

