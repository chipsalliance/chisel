; RUN: firrtl -i %s -o %s.v -X verilog -p cT 2>&1 | tee %s.out | FileCheck %s
circuit Top : 
  module Top : 
    node x = UInt("h2")
    node y = UInt("h100000")
    node z = UInt("h00")

;CHECK: node x = UInt<2>("h2")
;CHECK: node z = UInt<1>("h0")
;CHECK: Done!
