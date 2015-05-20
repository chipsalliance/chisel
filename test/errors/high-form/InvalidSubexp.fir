; RUN: firrtl -i %s -o %s.flo -X flo -p c | tee %s.out | FileCheck %s
; CHECK: Invalid index access to non-reference.
; CHECK: Invalid subfield access to non-reference.

circuit Top :
  module Top :
    wire x : UInt<4>
    add(x,x)[10] := UInt(1)
    add(x,x).x := UInt(1)
