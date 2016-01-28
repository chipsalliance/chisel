; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
; CHECK: Expand Whens
circuit top :
   module top :
      input clk : Clock
      input reset : UInt<1>
      wire p : UInt
      wire q : UInt
      wire a : UInt
      wire b : UInt
      wire x : UInt
      wire y : UInt
      wire z : UInt
      wire w : UInt
      reg r : UInt, clk with :
         reset => (reset, w)
      p <= UInt(1)
      q <= UInt(1)
      a <= UInt(1)
      b <= UInt(1)
      x <= UInt(1)
      y <= UInt(1)
      z <= UInt(1)
      w <= UInt(1)

      when p :
        r <= a
      when q :
        r <= b
      r <= z
; CHECK: r <= z
; CHECK: Finished Expand Whens
; CHECK: Done!
