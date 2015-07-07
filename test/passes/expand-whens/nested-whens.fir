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
      p := UInt(1)
      q := UInt(1)
      a := UInt(1)
      b := UInt(1)
      x := UInt(1)
      y := UInt(1)
      z := UInt(1)
      w := UInt(1)

      on-reset r := w
      when p :
        on-reset r := x
        r := a
      when q :
        on-reset r := y
        r := b
      r := z
; CHECK: when UInt(1) : r := mux(reset, mux(q, y, mux(p, x, w)), z)
; CHECK: Finished Expand Whens
