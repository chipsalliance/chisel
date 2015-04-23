; RUN: firrtl -i %s -o %s.flo -x abcd -p ck | tee %s.out | FileCheck %s

; CHECK: Resolve Kinds
circuit top :
   module subtracter :
      input x : UInt
      input y : UInt
      output z : UInt
      z := sub-wrap(x, y)
      ;CHECK: z@<k:port> := sub-wrap(x@<k:port>, y@<k:port>)
   module gcd :
      input a : UInt<16>
      input b : UInt<16>
      input e : UInt<1>
      output z : UInt<16>
      output v : UInt<1>
      reg x : UInt
      reg y : UInt
      on-reset x := UInt(0)
      on-reset y := UInt(42)
      when gt(x, y) :
         inst s of subtracter
         s.x := x
         ;CHECK: s@<k:inst>.x := x@<k:reg>
         s.y := y
         x := s.z
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
      ;CHECK: inst i of gcd@<k:module>
      i.a := a
      i.b := b
      i.e := UInt(1)
      z := i.z
      ;CHECK: z@<k:port> := i@<k:inst>.z
; CHECK: Finished Resolve Kinds


