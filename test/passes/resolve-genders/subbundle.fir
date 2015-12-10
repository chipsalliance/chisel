; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s

;CHECK: Lower To Ground
circuit top :
   module top :
      input clk : Clock
      input reset : UInt<1>
      wire w : { flip x : UInt<10>}
      reg r : { flip x : UInt<10>},clk,reset
      w <= r     ; CHECK r$x := w$x
      w.x <= r.x ; CHECK w$x := r$x
; CHECK: Finished Lower To Ground

