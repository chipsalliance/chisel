; RUN: firrtl -i %s -o %s.flo -X flo -p cTwd | tee %s.out | FileCheck %s

;CHECK: Infer Widths
circuit top :
   module top :
      wire e : UInt<30>
      e := UInt(1)
      reg y : UInt
      y := e

      wire a : UInt<20>
      a := UInt(1)
      wire b : UInt<10>
      b := UInt(1)
      wire c : UInt
      c := UInt(1)
      wire z : UInt
       
      z := mux(c,a,b)

   

; CHECK: Finished Infer Widths

