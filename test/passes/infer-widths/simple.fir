; RUN: firrtl -i %s -o %s.flo -x abcdefghijkl -p cT | tee %s.out | FileCheck %s

;CHECK: Infer Widths
circuit top :
   module top :
      wire e : UInt(30)
      reg y : UInt
      y := e
   

; CHECK: Finished Infer Widths

