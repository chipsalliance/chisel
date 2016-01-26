; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s

; CHECK: Expand Whens
circuit top :
   module top :
      input clk : Clock
      input reset : UInt<1>
      poison x : UInt<1>
      wire init : UInt<1>[10]
      init[0] <= x
      init[1] <= x
      init[2] <= x
      init[3] <= UInt(3)
      init[4] <= x
      init[5] <= x
      init[6] <= x
      init[7] <= x
      init[8] <= x
      init[9] <= x
      reg r : UInt<1>[10],clk with :
         reset => (reset,init)
      r[0] <= UInt(1)
      r[1] <= UInt(1)
      r[2] <= UInt(1)
      r[3] <= UInt(1)
      r[4] <= UInt(1)
      r[5] <= UInt(1)
      r[6] <= UInt(1)
      r[7] <= UInt(1)
      r[8] <= UInt(1)
      r[9] <= UInt(1)

; CHECK: Finished Expand Whens
; CHECK: Done!
