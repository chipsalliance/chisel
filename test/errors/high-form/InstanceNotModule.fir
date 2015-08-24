; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
; CHECK: Module Top2 is not defined.

circuit Top :
  module Top :
    wire x : UInt<1>
    inst t of Top2

