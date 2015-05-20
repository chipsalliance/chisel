; RUN: firrtl -i %s -o %s.flo -X flo -p c | tee %s.out | FileCheck %s
; CHECK: Module Top2 is not defined.

circuit Top :
  module Top :
    wire x : UInt<1>
    inst t of Top2

