; RUN: firrtl -i %s -o %s.flo -x abcdefgh -p c | tee %s.out | FileCheck %s

; CHECK: Lower To Ground
         circuit top :
            module top :
               input a : UInt<16>
               input b : UInt<16>
               output z : UInt

               reg r1 : { x : UInt, flip y : SInt }
               wire q : { x : UInt, flip y : SInt }
               on-reset r1 := q    

      ; CHECK: reg r1$x : UInt
      ; CHECK: reg r1$y : SInt
      ; CHECK: wire q$x : UInt
      ; CHECK: wire q$y : SInt
      ; CHECK: r1$x.init := q$x
      ; CHECK: q$y := r1$y.init

; CHECK: Finished Lower To Ground
