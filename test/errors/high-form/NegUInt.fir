; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
; CHECK: UIntValue cannot be negative.

circuit Top :
  module Top :
    wire x : UInt<4>
    x <= UInt(-2)
