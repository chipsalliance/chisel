; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
; CHECK: Node must be a ground type.

circuit Top : 
  module Top : 
    wire x : {x : UInt, flip y : UInt}
    node z = x

