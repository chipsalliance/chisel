; RUN: firrtl -i %s -o %s.flo -X flo -p ct | tee %s.out | FileCheck %s

;CHECK: Infer Types
circuit top :
   module subtracter :
      input x : UInt
      input y : UInt
      output z : UInt
      z := sub-wrap(x, y)
      ;CHECK: z@<t:UInt> := sub-wrap(x@<t:UInt>, y@<t:UInt>)@<t:UInt>
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
      ;CHECK: when gt(x@<t:UInt>, y@<t:UInt>)@<t:UInt> :
         inst s of subtracter
         ;CHECK: inst s of subtracter@<t:{flip x : UInt@<t:UInt>, flip y : UInt@<t:UInt>,  z : UInt@<t:UInt>, flip reset : UInt<1>@<t:UInt>}>
         s.x := x
         s.y := y
         x := s.z
         ;CHECK: s@<t:{flip x : UInt@<t:UInt>, flip y : UInt@<t:UInt>,  z : UInt@<t:UInt>, flip reset : UInt<1>@<t:UInt>}>.reset@<t:UInt> := reset@<t:UInt>
         ;CHECK: s@<t:{flip x : UInt@<t:UInt>, flip y : UInt@<t:UInt>,  z : UInt@<t:UInt>, flip reset : UInt<1>@<t:UInt>}>.x@<t:UInt> := x@<t:UInt>
         ;CHECK: s@<t:{flip x : UInt@<t:UInt>, flip y : UInt@<t:UInt>,  z : UInt@<t:UInt>, flip reset : UInt<1>@<t:UInt>}>.y@<t:UInt> := y@<t:UInt>
         ;CHECK: x@<t:UInt> := s@<t:{flip x : UInt@<t:UInt>, flip y : UInt@<t:UInt>,  z : UInt@<t:UInt>, flip reset : UInt<1>@<t:UInt>}>.z@<t:UInt>      
      else :
         inst s2 of subtracter
         s2.x := x
         s2.y := y
         y := s2.z
      when e :
         x := a
         y := b
      v := eq(v, UInt(0))
      ;CHECK: v@<t:UInt> := eq(v@<t:UInt>, UInt(0))@<t:UInt>
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

; CHECK: Finished Infer Types
