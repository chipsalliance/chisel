; RUN: firrtl -i %s -o %s.flo -x abcdefghij -p c | tee %s.out | FileCheck %s
; CHECK: Done!
circuit top :
   module A :
      reg r : { x : UInt, flip y : UInt}
      wire a : UInt
      wire b : UInt
      wire w : { x : UInt, flip y : UInt}

      r.x := a
      r.y := b
      on-reset r := w
      
; CHECK: reg r : { x, flip y}
; CHECK: r.x := a
; CHECK: r.y := b
; CHECK: when reset :
; CHECK:    r.x := w.x
; CHECK:    w.y := r.y


