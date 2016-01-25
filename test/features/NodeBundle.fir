; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
;CHECK: Lower Types
circuit Top : 
  module Top : 
    input a: {w:UInt<42>,x:UInt<20>}
    node d = a
;CHECK: node d_w = a_w
;CHECK: node d_x = a_x

;CHECK: Finished Lower Types
;CHECK: Done!

