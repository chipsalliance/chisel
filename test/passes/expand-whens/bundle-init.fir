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

; CHECK: node r_x = Register(mux(reset, w_x, a), UInt(1))
; CHECK: node r_y = Register(b, UInt(1))
; CHECK: a := UInt(1)
; CHECK: b := UInt(2)
; CHECK: w_x := b
; CHECK: w_y := mux(reset, r_y, a)   
      
; CHECK: Finished Expand Whens
