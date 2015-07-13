; RUN: firrtl -i %s -o %s.v -X verilog -p c | tee %s.out | FileCheck %s
; CHECK: Done!

circuit Top :
  module Top :
    wire x : UInt<1>
    x := UInt(1)
    wire x~y : UInt<2>
    x~y := UInt(1)
    wire x!y : UInt<2>
    x!y := UInt(1)
    wire x@y : UInt<2>
    x@y := UInt(1)
    wire x#y : UInt<2>
    x#y := UInt(1)
    wire x%y : UInt<2>
    x%y := UInt(1)
    wire x^y : UInt<2>
    x^y := UInt(1)
    wire x*y : UInt<2>
    x*y := UInt(1)
    wire x-y : UInt<2>
    x-y := UInt(1)
    wire x_y : UInt<2>
    x_y := UInt(1)
    wire x+y : UInt<2>
    x+y := UInt(1)
    wire x=y : UInt<2>
    x=y := UInt(1)
    wire x?y : UInt<2>
    x?y := UInt(1)
    wire x/y : UInt<2>
    x/y := UInt(1)


