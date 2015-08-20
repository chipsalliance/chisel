; RUN: firrtl -i %s -o %s.flo -X flo -p c | tee %s.out | FileCheck %s
; CHECK: Poison p cannot be a bundle type with flips.

circuit Flip-Poison : 
  module Flip-Poison : 
    input clk : Clock
    poison p : {x : UInt<10>, flip y : UInt<10>}

