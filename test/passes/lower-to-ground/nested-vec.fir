; RUN: firrtl -i %s -o %s.flo -X flo -p c | tee %s.out | FileCheck %s

; CHECK: Lower To Ground
circuit top :
   module top :
      wire i : UInt
      wire j : { x : UInt<32>, flip y : UInt<32> }
      wire k : { x : UInt<32>, y : UInt<32> }

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

      cmem m : { x : UInt<32>, y : UInt<32> }[2]
      ; CHECK: cmem m$x : UInt<32>[2]
      ; CHECK: cmem m$y : UInt<32>[2]

      infer accessor c = m[i] ; MALE
      ; CHECK: accessor c$x = m$x[i]
      ; CHECK: accessor c$y = m$y[i]

      c := k
      ; CHECK: c$x := k$x
      ; CHECK: c$y := k$y

; CHECK: Finished Lower To Ground

