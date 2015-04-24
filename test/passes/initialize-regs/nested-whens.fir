; RUN: firrtl -i %s -o %s.flo -x abcdefghij -p c | tee %s.out | FileCheck %s
; CHECK: Done!
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

      on-reset r := w
      when p
        on-reset r := x
        r := a
      when q
        on-reset r := y
        r := b
      r := z
      
; CHECK: r := z
; CHECK: when reset
; CHECK:    r := q?b:(p?a:w)

