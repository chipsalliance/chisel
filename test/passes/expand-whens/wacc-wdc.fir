; RUN: firrtl -i %s -o %s.flo -X flo -p c | tee %s.out | FileCheck %s
circuit top :
   module top :
      input clk : Clock
      wire p : UInt
      cmem m : UInt<4>[10], clk
      p := UInt(1)
      when p :
         write accessor a = m[UInt(3)]
         a := UInt(20)

; CHECK: Expand Whens

; CHECK: circuit top :
; CHECK:   module top :
; CHECK:     wire p : UInt
; CHECK:     cmem m : UInt<4>[10]
; CHECK:     write accessor a : m[UInt(3)]
; CHECK:     p := UInt(1)
; CHECK:     when p : a := UInt(20)

; CHECK: Finished Expand Whens


