; RUN: firrtl -i %s -o %s.flo -X flo -p c | tee %s.out | FileCheck %s
; CHECK: Memory m cannot be a bundle type with flips.

circuit Flip-Mem : 
  module Flip-Mem : 
    mem m : {x : UInt<3>, flip y : UInt<5>}[10]
