; RUN: firrtl -i %s -o %s.flo -X flo -p c | tee %s.out | FileCheck %s
; XFAIL: *
circuit top :
   module top :
      input clk : Clock
      input reset : UInt<1>
      wire p : UInt
      p := UInt(1)
      when p :
         reg r : UInt,clk,reset
         r := UInt(2)

; CHECK: Expand Whens

; CHECK: circuit top :
; CHECK:   module top :
; CHECK:     wire p : UInt
; CHECK:     reg r : UInt, clk, reset
; CHECK:     p := UInt("h00000001")
; CHECK-NOT: when p : r := UInt("h00000002")

; CHECK: Finished Expand Whens

