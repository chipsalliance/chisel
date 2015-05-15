; RUN: firrtl -i %s -o %s.flo -X flo -p cd | tee %s.out | FileCheck %s
; CHECK: Expand Whens
circuit top :
   module top :
      reg r : { x : UInt, flip y : UInt}
      wire a : UInt
      wire b : UInt
      wire w : { x : UInt, flip y : UInt}
      a := UInt(1)
      b := UInt(2)

      w.x := b
      w.y := a
      r.x := a
      r.y := b
      on-reset r := w

; CHECK: r$x := Register(mux-uu(reset, w$x, a), UInt(1))
; CHECK: r$y := Register(b, UInt(1))
; CHECK: a := UInt(1)
; CHECK: b := UInt(2)
; CHECK: w$x := b
; CHECK: w$y := mux-uu(reset, r$y, a)   
      
; CHECK: Finished Expand Whens
