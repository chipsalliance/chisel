; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
; XFAIL: *
;CHECK: Done!
circuit Top : 
   module Top :
      input in : UInt<1>
      wire b : UInt<1>[3]
      b.0 := UInt(1)
      b.1 := UInt(1)
      b.2 := UInt(1)
      node c = UInt(1)
      infer accessor a = b[c]
      when in : 
         a := UInt(1)
