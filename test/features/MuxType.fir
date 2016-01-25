
; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
;CHECK: Expand Connects
circuit Top : 
  module Top : 
    input a: UInt<1>
    input b: UInt<32>
    input p : UInt<1>
    output d: UInt
    node y = mux(p, a, b)
    d <= y
;CHECK:Done!
