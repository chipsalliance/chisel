; RUN: firrtl -i %s -o %s.flo -X flo -p c 2>&1 | tee %s.out | FileCheck %s
circuit top :
   module top :
      input clk : Clock
      input reset : UInt<1>
      wire p : UInt
      p := UInt(1)
      when p :
         reg r : UInt, clk, reset
         onreset r := UInt(1)
         r := UInt(2)

; CHECK: Expand Whens

; CHECK: circuit top :
; CHECK:   module top :
; CHECK:   wire p : UInt
; CHECK:   reg r : UInt, clk, reset
; CHECK-NOT:  when p : r := mux(reset, UInt("h00000001"), UInt("h00000002"))

; CHECK: Finished Expand Whens
