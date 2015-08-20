; RUN: firrtl -i %s -o %s.flo -X flo -p c 2>&1 | tee %s.out | FileCheck %s
;CHECK: Done!
circuit Top : 
   module Top :
      input in : UInt<1>
      wire b : UInt<1>[3]
      node c = UInt(1)
      infer accessor a = b[c]
      when in : 
         a := UInt(1)
