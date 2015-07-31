; RUN: firrtl -i %s -o %s.flo -X flo -p ct 2>&1 | tee %s.out | FileCheck %s

;CHECK: Infer Types
circuit top :
   module subtracter :
      input x : UInt
      input y : UInt
      output z : UInt
      z := subw(x, y)
      ;CHECK: z@<t:UInt> := subw(x@<t:UInt>, y@<t:UInt>)@<t:UInt>
   module gcd :
      input a : UInt<16>
      input b : UInt<16>
      input e : UInt<1>
      input clk : Clock
      input reset : UInt<1>
      output z : UInt<16>
      output v : UInt<1>
      reg x : UInt,clk,reset
      reg y : UInt,clk,reset
; CHECK: reg x : UInt, clk@<t:Clock>, reset@<t:UInt>@<t:UInt>
      onreset x := UInt(0)
      onreset y := UInt(42)
      when gt(x, y) :
      ;CHECK: when gt(x@<t:UInt>, y@<t:UInt>)@<t:UInt> :
         inst s of subtracter
         ;CHECK: inst s of subtracter@<t:{flip x : UInt@<t:UInt>, flip y : UInt@<t:UInt>,  z : UInt@<t:UInt>}>
         s.x := x
         s.y := y
         x := s.z
         ;CHECK: s@<t:{flip x : UInt@<t:UInt>, flip y : UInt@<t:UInt>,  z : UInt@<t:UInt>}>.x@<t:UInt> := x@<t:UInt>
         ;CHECK: s@<t:{flip x : UInt@<t:UInt>, flip y : UInt@<t:UInt>,  z : UInt@<t:UInt>}>.y@<t:UInt> := y@<t:UInt>
         ;CHECK: x@<t:UInt> := s@<t:{flip x : UInt@<t:UInt>, flip y : UInt@<t:UInt>,  z : UInt@<t:UInt>}>.z@<t:UInt>      
      else :
         inst s2 of subtracter
         s2.x := x
         s2.y := y
         y := s2.z
      when e :
         x := a
         y := b
      z := x
   module top :
      input a : UInt<16>
      input b : UInt<16>
      input clk : Clock
      input reset : UInt<1>
      output z : UInt
      inst i of gcd
      i.a := a
      i.b := b
      i.clk := clk
      i.reset := reset
      i.e := UInt(1)
      z := i.z

; CHECK: Finished Infer Types
