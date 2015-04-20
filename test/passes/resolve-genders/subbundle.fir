; RUN: firrtl -i %s -o %s.flo -x abcdefgh -p c | tee %s.out | FileCheck %s

;CHECK: Lower To Ground
circuit top :
   module M :
      wire w : { flip x : UInt(10)}
      reg r : { flip x : UInt(10)}
      w := r     ; CHECK r$x := w$x
      w.x := r.x ; CHECK w$x := r$x
; CHECK: Finished Lower To Ground

