; RUN: firrtl %s abcde | tee %s.out | FileCheck %s

;CHECK: Infer Types
circuit top :
   module subtracter :
      input x : UInt
      input y : UInt
      output z : UInt
      z := sub-mod(x, y)
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
      when greater(x, y) :
         inst s of subtracter
         s.x := x
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
      v := equal(v, UInt(0))
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


