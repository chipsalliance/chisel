; RUN: firrtl -i %s -o %s.flo -X flo -p c | tee %s.out | FileCheck %s
circuit top :
   module top :
      input clk : Clock
      input reset : UInt<1>
      wire p : UInt
      p := UInt(1)
      when p :
         reg r : UInt,clk,reset
         r := UInt(20)

; CHECK: Expand Whens

; CHECK: circuit top :
; CHECK:   module top :
; CHECK:     wire p : UInt
; CHECK:     reg r : UInt
; CHECK:     p := UInt(1)
; CHECK:     r := UInt(20)

; CHECK: Finished Expand Whens

