; RUN: firrtl -i %s -o %s.flo -X flo -p cTwd | tee %s.out | FileCheck %s

;CHECK: Infer Widths
circuit top :
   module top :
      wire e : UInt<30>
      reg y : UInt
      y := e

      wire a : UInt<20>
      wire b : UInt<10>
      wire c : UInt
      wire z : UInt
       
      z := mux(c,a,b)

   

; CHECK: Finished Infer Widths

