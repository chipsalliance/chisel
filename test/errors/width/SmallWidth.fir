; RUN: firrtl -i %s -o %s.v -X verilog -p ciwT 2>&1 | tee %s.out | FileCheck %s
; CHECK: Width too small for UIntValue.

circuit Top : 
  module Top : 
    output z : UInt

    z <= add(UInt<4>("h121"),UInt<3>("h13333"))




