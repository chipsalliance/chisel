; RUN: firrtl -i %s -o %s.v -X verilog ; cat %s.v | FileCheck %s

;CHECK: reg  [29:0] m [0:127];
circuit top :
   module top :
      input clk : Clock
      output read : UInt<30>
      cmem m : UInt<30>[128], clk
      read accessor x = m[UInt(0)]
      read <= x
      

