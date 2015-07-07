; RUN: firrtl -i %s -o %s.flo -X flo -p c | tee %s.out | FileCheck %s
; CHECK: Invalid connect to an expression that is not a reference or a WritePort.
; CHECK: Invalid connect to an expression that is not a reference or a WritePort.
; CHECK: Invalid connect to an expression that is not a reference or a WritePort.

circuit Top :
  module Top :
    wire x : UInt
    add(x,x) := UInt(1)
    UInt(1) := UInt(1)
    SInt(1) := UInt(1)

