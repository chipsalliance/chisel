; RUN: firrtl -i %s -o %s.v -X verilog -p c | tee %s.out | FileCheck %s
; CHECK: Done!

circuit Test : 
  module Test : 
     wire x : UInt<10>
     x := UInt(5151)
     x := UInt(515151512512412414124124124124)

