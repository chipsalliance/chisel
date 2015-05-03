; RUN: firrtl -i %s -o %s.flo -x abcdefghijkl -p cd | tee %s.out | FileCheck %s

;CHECK: Infer Widths

circuit top :
   module M :
      wire x : UInt<16>
      wire z : SInt<16>
      wire y : UInt<3>
      wire a : UInt
      wire b : SInt
      wire c : UInt
      wire d : SInt

      a := dshl-u(x,y)
      b := dshl-s(z,y)
      c := dshr-u(x,y)
      d := dshr-s(z,y)


; CHECK: wire a : UInt<23>
; CHECK: wire b : SInt<23>
; CHECK: wire c : UInt<16>
; CHECK: wire d : SInt<16>
; CHECK: Finished Infer Widths
