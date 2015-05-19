; RUN: firrtl -i %s -o %s.flo -X flo -p cg | tee %s.out | FileCheck %s

;CHECK: Resolve Genders
circuit top :
   module subtracter :
      input x : UInt
      input y : UInt
      output z : UInt
      z := sub-wrap(x, y)
      ;CHECK: z@<g:f> := sub-wrap(x@<g:m>, y@<g:m>)
   module gcd :
      input a : UInt<16>
      input b : UInt<16>
      input e : UInt<1>
      output z : UInt<16>
      output v : UInt<1>
      reg x : UInt
      reg y : UInt
; CHECK: reg x : UInt
      on-reset x := UInt(0)
      on-reset y := UInt(42)
      when gt(x, y) :
      ;CHECK: when gt(x@<g:m>, y@<g:m>) :
         inst s of subtracter
         ;CHECK: inst s of subtracter@<g:m>
         s.x := x
         s.y := y
         x := s.z
         ;CHECK: s@<g:m>.x@<g:f> := x@<g:m>
         ;CHECK: s@<g:m>.y@<g:f> := y@<g:m>
         ;CHECK: x@<g:f> := s@<g:m>.z@<g:m>
      else :
         inst s2 of subtracter
         s2.x := x
         s2.y := y
         y := s2.z
      when e :
         x := a
         y := b
      v := eq(v, UInt(0))
      z := x
   module top :
      input a : UInt<16>
      input b : UInt<16>
      output z : UInt
      inst i of gcd
      i.a := a
      i.b := b
      i.e := UInt(1)
      z := i.z

; CHECK: Finished Resolve Genders
