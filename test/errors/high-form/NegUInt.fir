; RUN: firrtl -i %s -o %s.flo -x X -p c | tee %s.out | FileCheck %s
; CHECK: UIntValue cannot be negative.

circuit Top :
  module Top :
    wire x : UInt<4>
    x := UInt(-2)
