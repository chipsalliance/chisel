; RUN: firrtl -i %s -o %s.v -X verilog -p c | tee %s.out | FileCheck %s
; CHECK: Done!

circuit Top :
  module Top :
    wire x : UInt<1>
    node p = UInt(1)
    when p :
       wire x : UInt<1>
       x := UInt(1)
       node y = add(x,UInt(1))
    else :
       wire x : UInt<1>
       x := UInt(1)
       node z = add(x,UInt(1))
    x := UInt(1)
    node w = add(x,UInt(1))
