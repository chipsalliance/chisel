; RUN: firrtl -i %s -o %s.flo -X flo -p c | tee %s.out | FileCheck %s
; CHECK: Expand Whens
circuit top :
   module top :
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
; CHECK: node r = Register(mux(reset, mux(q, y, mux(p, x, w)), z), UInt(1))
; CHECK: Finished Expand Whens
