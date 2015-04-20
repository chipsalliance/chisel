; RUN: firrtl -i %s -o %s.flo -x abcde -p ct | tee %s.out | FileCheck %s

;CHECK: Infer Types
circuit top :
   module subtracter :
      input x : UInt
      input y : UInt
      output z : UInt
      z := sub-wrap(x, y)
      ;CHECK: z@<t:UInt> := sub-wrap-uu(x@<t:UInt>, y@<t:UInt>)@<t:UInt>
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
      ;CHECK: when gt-uu(x@<t:UInt>, y@<t:UInt>)@<t:UInt> :
         inst s of subtracter
         ;CHECK: inst s of subtracter@<t:{ x : UInt@<t:UInt>, y : UInt@<t:UInt>, flip z : UInt@<t:UInt>, reset : UInt(1)@<t:UInt>}>
         s.x := x
         s.y := y
         x := s.z
         ;CHECK: s@<t:{ x : UInt@<t:UInt>, y : UInt@<t:UInt>, flip z : UInt@<t:UInt>, reset : UInt(1)@<t:UInt>}>.reset@<t:UInt> := reset@<t:UInt>
         ;CHECK: s@<t:{ x : UInt@<t:UInt>, y : UInt@<t:UInt>, flip z : UInt@<t:UInt>, reset : UInt(1)@<t:UInt>}>.x@<t:UInt> := x@<t:UInt>
         ;CHECK: s@<t:{ x : UInt@<t:UInt>, y : UInt@<t:UInt>, flip z : UInt@<t:UInt>, reset : UInt(1)@<t:UInt>}>.y@<t:UInt> := y@<t:UInt>
         ;CHECK: x@<t:UInt> := s@<t:{ x : UInt@<t:UInt>, y : UInt@<t:UInt>, flip z : UInt@<t:UInt>, reset : UInt(1)@<t:UInt>}>.z@<t:UInt>
      else :
         inst s2 of subtracter
         s2.x := x
         s2.y := y
         y := s2.z
      when e :
         x := a
         y := b
      v := eq(v, UInt(0))
      ;CHECK: v@<t:UInt> := eq-uu(v@<t:UInt>, UInt(0))@<t:UInt>
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
