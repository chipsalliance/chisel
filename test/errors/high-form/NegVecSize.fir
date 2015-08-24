; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
; CHECK: Vector type size cannot be negative

circuit Top : 
  module Top : 
    wire x : UInt<3>[-5]

