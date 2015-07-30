; RUN: firrtl -i %s -o %s.flo -X flo -p c | tee %s.out | FileCheck %s
; CHECK: Memory mc cannot be a bundle type with flips.
; CHECK: Memory ms cannot be a bundle type with flips.

circuit Flip-Mem : 
  module Flip-Mem : 
    input clk : Clock
    cmem mc : {x : UInt<3>, flip y : UInt<5>}[10], clk
    smem ms : {x : UInt<3>, flip y : UInt<5>}[10], clk
