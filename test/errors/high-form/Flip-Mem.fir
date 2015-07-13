; RUN: firrtl -i %s -o %s.flo -X flo -p c | tee %s.out | FileCheck %s
; CHECK: Memory m-c cannot be a bundle type with flips.
; CHECK: Memory m-s cannot be a bundle type with flips.

circuit Flip-Mem : 
  module Flip-Mem : 
    input clk : Clock
    cmem m-c : {x : UInt<3>, flip y : UInt<5>}[10], clk
    smem m-s : {x : UInt<3>, flip y : UInt<5>}[10], clk
