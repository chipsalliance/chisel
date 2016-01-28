; RUN: firrtl -i %s -o %s.v -X verilog -p cg 2>&1 | tee %s.out | FileCheck %s

;CHECK: Resolve Genders
circuit top :
   module subtracter :
      input x : UInt
      input y : UInt
      output z : UInt
      z <= tail(sub(x, y),1)
      ;CHECK: z@<g:f> <= tail(sub(x@<g:m>, y@<g:m>)@<g:m>, 1)@<g:m>   
   module gcd :
      input a : UInt<16>
      input b : UInt<16>
      input e : UInt<1>
      input clk : Clock
      input reset : UInt<1>
      output z : UInt<16>
      output v : UInt<1>
      reg x : UInt,clk with :
         reset => (reset,UInt(0))
      reg y : UInt,clk with :
         reset => (reset,UInt(42))
; CHECK: reg x : UInt
      when gt(x, y) :
      ;CHECK: when gt(x@<g:m>, y@<g:m>)@<g:m> :
         inst s of subtracter
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
