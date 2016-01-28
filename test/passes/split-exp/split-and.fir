; RUN: firrtl -i %s -o %s.v -X verilog -p cT 2>&1 | tee %s.out | FileCheck %s

; CHECK: Done!
circuit Top :
   module Top :
      input a : SInt<2>
      output c : UInt<2>
      c <= and(a,asSInt(UInt(2)))
