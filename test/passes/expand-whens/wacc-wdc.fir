; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
circuit top :
   module top :
      input clk : Clock
      wire p : UInt
      cmem m : UInt<4>[10], clk
      p := UInt(1)
      when p :
         write accessor a = m[UInt(3)]
         a := UInt(2)

; CHECK: Expand Whens

; CHECK: circuit top :
; CHECK:   module top :
; CHECK:     wire p : UInt
; CHECK:     cmem m : UInt<4>[10], clk
; CHECK:     write accessor a = m[UInt("h3")]
; CHECK:     p := UInt("h1")
; CHECK:     when p : a := UInt("h2")

; CHECK: Finished Expand Whens


