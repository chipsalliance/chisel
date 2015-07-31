; RUN: firrtl -i %s -o %s.flo -X flo -p c 2>&1 | tee %s.out | FileCheck %s
; XFAIL: *
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
; CHECK:     write accessor a = m[UInt("h00000003")]
; CHECK:     p := UInt("h00000001")
; CHECK:     when p : a := UInt("h00000002")

; CHECK: Finished Expand Whens


