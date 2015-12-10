; RUN: firrtl -i %s -o %s.v -X verilog -p cg 2>&1 | tee %s.out | FileCheck %s

;CHECK: Resolve Genders
circuit top :
   module subtracter :
      input x : UInt
      input y : UInt
      output z : UInt
      z <= subw(x, y)
      ;CHECK: z@<g:f> <= subw(x@<g:m>, y@<g:m>)
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
; CHECK: reg x : UInt
      onreset x <= UInt(0)
      onreset y <= UInt(42)
      when gt(x, y) :
      ;CHECK: when gt(x@<g:m>, y@<g:m>) :
         inst s of subtracter
         ;CHECK: inst s of subtracter@<g:m>
         s.x <= x
         s.y <= y
         x <= s.z
         ;CHECK: s@<g:m>.x@<g:f> <= x@<g:m>
         ;CHECK: s@<g:m>.y@<g:f> <= y@<g:m>
         ;CHECK: x@<g:f> <= s@<g:m>.z@<g:m>
      else :
         inst s2 of subtracter
         s2.x <= x
         s2.y <= y
         y <= s2.z
      when e :
         x <= a
         y <= b
      v <= eq(v, UInt(0))
      z <= x
   module top :
      input clk : Clock
      input reset : UInt<1>
      input a : UInt<16>
      input b : UInt<16>
      output z : UInt
      inst i of gcd
      i.a <= a
      i.b <= b
      i.clk <= clk
      i.reset <= reset
      i.e <= UInt(1)
      z <= i.z

; CHECK: Finished Resolve Genders
