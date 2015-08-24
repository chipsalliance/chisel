; RUN: firrtl -i %s -o %s.flo -X flo -p c 2>&1 | tee %s.out | FileCheck %s
; XFAIL: *

;CHECK: Flo
circuit top :
   module subtracter :
      input x : UInt
      input y : UInt
      output q : UInt
      q := subw(x, y)
   module gcd :
      input clk : Clock
      input reset : UInt<1>
      input a : UInt<16>
      input b : UInt<16>
      input e : UInt<1>
      output z : UInt<16>
      reg x : UInt,clk,reset
      reg y : UInt,clk,reset
      onreset x := UInt(0)
      onreset y := UInt(42)
      when gt(x, y) :
         inst s of subtracter
         s.x := x
         s.y := y
         x := s.q
      else :
         inst s2 of subtracter
         s2.x := x
         s2.y := y
         y := s2.q
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
      i.clk := clk
      i.reset := reset
      i.a := a
      i.b := b
      i.e := UInt(1)
      z := i.z
;CHECK: Done!

