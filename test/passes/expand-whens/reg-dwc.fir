; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
circuit top :
   module top :
      input clk : Clock
      input reset : UInt<1>
      wire p : UInt
      p <= UInt(1)
      reg r : UInt,clk,reset,r
      when p :
         r <= UInt(2)

; CHECK: Expand Whens

; CHECK: circuit top :
; CHECK:   module top :
; CHECK:     wire p : UInt
; CHECK:     reg r : UInt
; CHECK:     p <= UInt("h1")
; CHECK:     r <= mux(p, UInt("h2"), r)

; CHECK: Finished Expand Whens

; CHECK: Done!

