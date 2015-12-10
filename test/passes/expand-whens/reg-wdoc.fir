; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
circuit top :
   module top :
      input clk : Clock
      input reset : UInt<1>
      wire p : UInt
      p <= UInt(1)
      when p :
         reg r : UInt,clk,reset
         onreset r <= UInt(1)
         r <= UInt(2)

; CHECK: Expand Whens

; CHECK: circuit top :
; CHECK:   module top :
; CHECK:     wire p : UInt
; CHECK:     reg r : UInt, clk, reset
; CHECK:     p <= UInt("h1")
; CHECK-NOT:  when p : r <= mux(reset, UInt("h1"), UInt("h2"))

; CHECK: Finished Expand Whens

