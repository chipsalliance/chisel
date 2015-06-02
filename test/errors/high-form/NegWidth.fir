; RUN: firrtl -i %s -o %s.flo -X flo -p c | tee %s.out | FileCheck %s
; CHECK: Width cannot be negative.

circuit Top : 
  module Top : 
    wire x : UInt<-3>

