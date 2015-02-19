; RUN: firrtl %s ab | tee %s.out | FileCheck %s

circuit top :
   module subtracter :
      input x : UInt
      input y : UInt
      output z : UInt
      z := sub-mod(x, y)
; CHECK: output z : UInt
; CHECK: port:z := sub-mod(port:x, port:y)

