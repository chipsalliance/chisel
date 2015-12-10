; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s

; CHECK: Expand Whens
circuit top :
   module top :
      input clk : Clock
      input reset : UInt<1>
      reg r : UInt<1>[10],clk,reset
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
      onreset r[3] <= UInt(0)

; CHECK: Finished Expand Whens
