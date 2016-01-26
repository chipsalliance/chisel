; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
; CHECK: Done!

circuit Top :
  module Top :
    wire x_1 : UInt<1>
    x_1 <= UInt(1)
    wire x : UInt<1>
    x <= add(add(UInt(1),UInt(1)),UInt(1))



