; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
; CHECK: Done!
circuit top :
   module top :
      input clk : Clock
      wire p : UInt
      wire q : UInt
      cmem m : {a:UInt<4>,b:{c: UInt<4>,d:UInt<4>}}[10]
      p <= UInt(1)
      q <= UInt(1)
      wire x : {a:UInt<4>,b:{c: UInt<4>,d:UInt<4>}}
      x.a <= UInt(1)
      x.b.c <= UInt(1)
      x.b.d <= UInt(1)
      when p :
         write mport a = m[UInt(3)],clk
         when q :
            a <- x


