; RUN: firrtl -i %s -o %s.flo -x abcdef -p cg | tee %s.out | FileCheck %s

;CHECK: Resolve Genders
circuit top :
   module subtracter :
      input x : UInt
      input y : UInt
      output z : UInt
      z := sub-wrap(x, y)
      ;CHECK: z@<g:female> := sub-wrap-uu(x@<g:male>, y@<g:male>)
   module gcd :
      input a : UInt(16)
      input b : UInt(16)
      input e : UInt(1)
      output z : UInt(16)
      output v : UInt(1)
      reg x : UInt
      reg y : UInt
; CHECK: reg x : UInt
      x.init := UInt(0)
      y.init := UInt(42)
      when gt(x, y) :
      ;CHECK: when gt-uu(x@<g:male>, y@<g:male>) :
         inst s of subtracter
         ;CHECK: inst s of subtracter@<g:female>
         s.x := x
         s.y := y
         x := s.z
         ;CHECK: s@<g:female>.x@<g:female> := x@<g:male>
         ;CHECK: s@<g:female>.y@<g:female> := y@<g:male>
         ;CHECK: x@<g:female> := s@<g:female>.z@<g:male>
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
      input a : UInt(16)
      input b : UInt(16)
      output z : UInt
      inst i of gcd
      i.a := a
      i.b := b
      i.e := UInt(1)
      z := i.z

; CHECK: Finished Resolve Genders
