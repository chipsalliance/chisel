; RUN: firrtl -i %s -o %s.v -X verilog -p cT 2>&1 | tee %s.out | FileCheck %s

;CHECK: Lower Types
circuit top :
   module top :
      input clk : Clock
      input reset : UInt<1>
      wire w : { flip x : UInt<10>}
      reg r : { flip x : UInt<10>},clk,reset,w
      w <= r     ; CHECK r_x := w_x
      w.x <= r.x ; CHECK w_x := r_x
; CHECK: Finished Lower Types

