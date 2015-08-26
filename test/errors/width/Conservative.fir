; RUN: firrtl -i %s -o %s.v -X verilog -p ciwT 2>&1 | tee %s.out | FileCheck %s
; CHECK: Width too small for UIntValue.

circuit Top : 
  module Top : 
    node bar = UInt<1>("h1111")
