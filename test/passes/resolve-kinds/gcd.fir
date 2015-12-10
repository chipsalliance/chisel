; RUN: firrtl -i %s -o %s.v -X verilog -p ck 2>&1 | tee %s.out | FileCheck %s

; CHECK: Resolve Kinds
circuit top :
   module subtracter :
      input x : UInt
      input y : UInt
      output z : UInt
      z <= subw(x, y)
      ;CHECK: z@<k:port> <= subw(x@<k:port>, y@<k:port>)
   module gcd :
      input clk : Clock
      input reset : UInt<1>
      input a : UInt<16>
      input b : UInt<16>
      input e : UInt<1>
      output z : UInt<16>
      output v : UInt<1>
      reg x : UInt,clk,reset
      reg y : UInt,clk,reset
      onreset x <= UInt(0)
      onreset y <= UInt(42)
      when gt(x, y) :
         inst s of subtracter
         s.x <= x
         ;CHECK: s@<k:inst>.x <= x@<k:reg>
         s.y <= y
         x <= s.z
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
      input a : UInt<16>
      input b : UInt<16>
      input clk : Clock
      input reset : UInt<1>
      output z : UInt
      inst i of gcd
      ;CHECK: inst i of gcd@<k:module>
      i.a <= a
      i.b <= b
      i.clk <= clk
      i.reset <= reset
      i.e <= UInt(1)
      z <= i.z
      ;CHECK: z@<k:port> <= i@<k:inst>.z
; CHECK: Finished Resolve Kinds


