; RUN: firrtl -i %s -o %s.flo -X flo -p c | tee %s.out | FileCheck %s

; CHECK: Lower To Ground
circuit top :
   module top :
      wire i : UInt
      wire j : { x : UInt<32>, flip y : UInt<32> }

      wire a : { x : UInt<32>, flip y : UInt<32> }[2]
      ; CHECK: wire a$0$x : UInt<32>
      ; CHECK: wire a$0$y : UInt<32>
      ; CHECK: wire a$1$x : UInt<32>
      ; CHECK: wire a$1$y : UInt<32>

      infer accessor b = a[i]
      ; CHECK: wire b$x : UInt<32>
      ; CHECK: wire b$y : UInt<32>
      ; CHECK: b$x := (a$0$x a$1$x)[i]
      ; CHECK: (a$0$y a$1$y)[i] := b$y
      j := b
      
      infer accessor c = a[i]
      ; CHECK: wire c$x : UInt<32>
      ; CHECK: wire c$y : UInt<32>
      ; CHECK: (a$0$x a$1$x)[i] := c$x
      ; CHECK: c$y := (a$0$y a$1$y)[i]
      c := j


; CHECK: Finished Lower To Ground

