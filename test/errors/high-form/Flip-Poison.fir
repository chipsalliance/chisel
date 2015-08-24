; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
; CHECK: Poison p cannot be a bundle type with flips.

circuit Flip-Poison : 
  module Flip-Poison : 
    input clk : Clock
    poison p : {x : UInt<10>, flip y : UInt<10>}

