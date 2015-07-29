; RUN: firrtl -i %s -o %s.v -X verilog -p c | tee %s.out | FileCheck %s
; CHECK: Done!

circuit Test : 
  module Test : 
     wire x : UInt
     x := UInt(0)
     x := UInt(1)
     x := UInt(10)
     x := UInt(21474836)
     x := UInt("h21474836")

