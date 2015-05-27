; RUN: firrtl -i %s -o %s.flo -X flo -p c | tee %s.out | FileCheck %s
; CHECK: Node cannot be a bundle type with flips.

circuit Top : 
  module Top : 
    wire x : {x : UInt, flip y : UInt}
    node z = x

