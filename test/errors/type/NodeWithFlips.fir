; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
; CHECK: Node cannot be a bundle type with flips.

circuit Top : 
  module Top : 
    wire x : {x : UInt, flip y : UInt}
    node z = x

