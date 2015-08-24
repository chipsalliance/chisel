; RUN: firrtl -i %s -o %s.v -X verilog -p cd 2>&1 | tee %s.out | FileCheck %s
; CHECK: Expand Whens
circuit top :
   module top :
      input clk : Clock
      input reset : UInt<1>
      reg r : { x : UInt, y : UInt},clk,reset
      wire a : UInt
      wire b : UInt
      wire w : { x : UInt, y : UInt}
      a := UInt(1)
      b := UInt(2)

      w.x := b
      w.y := a
      r.x := a
      r.y := b
      onreset r := w

; CHECK: r$x := mux(reset, w$x, a)
; CHECK: r$y := mux(reset, w$y, b)
; CHECK: a := UInt("h1")
; CHECK: b := UInt("h2")
; CHECK: w$x := b
; CHECK: w$y := a
      
; CHECK: Finished Expand Whens
