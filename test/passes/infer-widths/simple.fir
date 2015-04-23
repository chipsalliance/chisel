; RUN: firrtl -i %s -o %s.flo -x abcdefghijkl -p cTwd | tee %s.out | FileCheck %s

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
       
      z := mux(c,Pad(a,?),Pad(b,?))

   

; CHECK: Finished Infer Widths

