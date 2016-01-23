; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
; CHECK: Done!
circuit top :
   module top :
      input clk : Clock
      wire p : UInt
      wire q : UInt
      cmem m : {a:UInt<4>,b:UInt<4>}[10]
      p <= UInt(1)
      q <= UInt(1)
      wire x : {a:UInt<4>,b:UInt<4>}
      x.a <= UInt(1)
      x.b <= UInt(1)
      when p :
         infer mport a = m[UInt(3)],clk
         infer mport b = m[UInt(3)],clk
         infer mport c = m[UInt(3)],clk
         when q :
            a <= x
            x <= b
            c <= x
            x <= c
