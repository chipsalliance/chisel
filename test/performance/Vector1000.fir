; RUN: firrtl -i %s -o %s.v -X verilog 2>&1 | tee %s.out | FileCheck %s
; CHECK: Done!
circuit Top :
   module Top :
      input in1 : UInt<32>[1000]
      output out : UInt<32>[1000]
      out <= in1
