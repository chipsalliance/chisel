; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
; CHECK: Invalid access to non-reference.
; CHECK: Invalid access to non-reference.

circuit Top :
  module Top :
    wire x : UInt<4>
    add(x,x)[10] <= UInt(1)
    add(x,x).x <= UInt(1)
