; RUN: firrtl %s abcefghipjk cT | tee %s.out | FileCheck %s

;CHECK: Infer Widths
circuit top :
   module top :
      wire e : UInt
      wire x : UInt
      reg y : UInt
      y := mux-uu(e, UInt(1), equal-uu(gt-uu(x, x), UInt(0)))
   

; CHECK: Finished Infer Widths

