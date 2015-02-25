; RUN: firrtl %s abcde ct | tee %s.out | FileCheck %s

;CHECK: Infer Types
circuit top :
   module subtracter :
      input x : UInt
      input y : UInt
      output z : UInt
      z := sub-mod(x, y)
      ;CHECK: z@<t:UInt> := sub-mod(x@<t:UInt>, y@<t:UInt>)@<t:UInt>
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
      ;CHECK: when greater(x@<t:UInt>, y@<t:UInt>)@<t:UInt> :
         inst s of subtracter
         ;CHECK: inst s of subtracter@<t:{input x : UInt@<t:UInt>, input y : UInt@<t:UInt>, output z : UInt@<t:UInt>, input reset : UInt(1)@<t:UInt(1)>}>
         s.x := x
         s.y := y
         x := s.z
         ;CHECK: s@<t:{input x : UInt@<t:UInt>, input y : UInt@<t:UInt>, output z : UInt@<t:UInt>, input reset : UInt(1)@<t:UInt(1)>}>.reset@<t:UInt(1)> := reset@<t:UInt(1)>
         ;CHECK: s@<t:{input x : UInt@<t:UInt>, input y : UInt@<t:UInt>, output z : UInt@<t:UInt>, input reset : UInt(1)@<t:UInt(1)>}>.x@<t:UInt> := x@<t:UInt>
         ;CHECK: s@<t:{input x : UInt@<t:UInt>, input y : UInt@<t:UInt>, output z : UInt@<t:UInt>, input reset : UInt(1)@<t:UInt(1)>}>.y@<t:UInt> := y@<t:UInt>
         ;CHECK: x@<t:UInt> := s@<t:{input x : UInt@<t:UInt>, input y : UInt@<t:UInt>, output z : UInt@<t:UInt>, input reset : UInt(1)@<t:UInt(1)>}>.z@<t:UInt>
      else :
         inst s2 of subtracter
         s2.x := x
         s2.y := y
         y := s2.z
      when e :
         x := a
         y := b
      v := equal(v, UInt(0))
      ;CHECK: v@<t:UInt(1)> := equal(v@<t:UInt(1)>, UInt(0))@<t:UInt>
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

; CHECK: Finished Infer Types
