; RUN: firrtl -i %s -o %s.flo -x abcdefghijk -p c | tee %s.out | FileCheck %s
; CHECK: Expand Whens
circuit top :
   module A :
      wire p : UInt
      wire q : UInt
      reg r : UInt
      wire a : UInt
      wire b : UInt
      wire x : UInt
      wire y : UInt
      wire z : UInt
      wire w : UInt

      on-reset r := w
      when p :
        on-reset r := x
        r := a
      when q :
        on-reset r := y
        r := b
      r := z
; CHECK: r := Register(mux-uu(reset, mux-uu(q, y, mux-uu(p, x, w)), z), UInt(1))
; CHECK: Finished Expand Whens
