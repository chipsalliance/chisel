; RUN: firrtl -i %s -o %s.v -X verilog -p ciwTd 2>&1 | tee %s.out | FileCheck %s
; CHECK: Uninferred width.

circuit Top : 
  module Top : 
    input z : UInt
    





