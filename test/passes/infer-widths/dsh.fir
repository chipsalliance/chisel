; RUN: firrtl -i %s -o %s.v -X verilog -p cd 2>&1 | tee %s.out | FileCheck %s

;CHECK: Infer Widths

circuit top :
   module top :
      wire x : UInt<16>
      wire z : SInt<16>
      wire y : UInt<3>
      wire a : UInt
      wire b : SInt
      wire c : UInt
      wire d : SInt
      x := UInt(1)
      y := UInt(1)
      z := SInt(1)

      a := dshl(x,y)
      b := dshl(z,y)
      c := dshr(x,y)
      d := dshr(z,y)


; CHECK: wire a : UInt<23>
; CHECK: wire b : SInt<23>
; CHECK: wire c : UInt<16>
; CHECK: wire d : SInt<16>
; CHECK: Finished Infer Widths
