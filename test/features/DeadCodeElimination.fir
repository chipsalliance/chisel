; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
; XFAIL: *

circuit top :
   module top :
      input clk : Clock
      input reset : UInt<1>
      wire x : UInt<1>

; CHECK: Done!
