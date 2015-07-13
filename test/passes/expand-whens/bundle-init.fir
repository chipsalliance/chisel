; RUN: firrtl -i %s -o %s.flo -X flo -p cd | tee %s.out | FileCheck %s
; CHECK: Expand Whens
circuit top :
   module top :
      input clk : Clock
      input reset : UInt<1>
      reg r : { x : UInt, flip y : UInt},clk,reset
      wire a : UInt
      wire b : UInt
      wire w : { x : UInt, flip y : UInt}
      a := UInt(1)
      b := UInt(2)

      w.x := b
      w.y := a
      r.x := a
      r.y := b
      onreset r := w

; CHECK: when UInt(1) : r$x := mux(reset, w$x, a)
; CHECK: when UInt(1) : r$y := b
; CHECK: a := UInt(1)
; CHECK: b := UInt(2)
; CHECK: w$x := b
; CHECK: w$y := mux(reset, r$y, a)   
      
; CHECK: Finished Expand Whens
