; RUN: firrtl -i %s -o %s.v -X verilog -p c | tee %s.out | FileCheck %s
circuit Top : 
  module Top : 
    output z : UInt<4>
    inst i of BlackBox
    i.x := UInt(1)
    i.y := UInt(2)
    z := i.z
  extmodule BlackBox :
    input x : UInt<4>
    input y : UInt<4>
    output z : UInt<4>

;CHECK: Done!
