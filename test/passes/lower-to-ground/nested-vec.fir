; RUN: firrtl -i %s -o %s.flo -x abcdefgh -p c | tee %s.out | FileCheck %s

; CHECK: Lower To Ground
circuit top :
   module q :
      wire i : UInt
      wire j : { x : UInt<32>, flip y : UInt<32> }

      wire a : { x : UInt<32>, flip y : UInt<32> }[2]
      ; CHECK: wire a$0$x : UInt<32>
      ; CHECK: wire a$0$y : UInt<32>
      ; CHECK: wire a$1$x : UInt<32>
      ; CHECK: wire a$1$y : UInt<32>

      accessor b = a[i]
      ; CHECK: wire b$x : UInt<32>
      ; CHECK: wire b$y : UInt<32>
      ; CHECK: b$x := (a$0$x a$1$x)[i]
      ; CHECK: (a$0$y a$1$y)[i] := b$y
      j := b

      mem m : { x : UInt<32>, flip y : UInt<32> }[2]
      ; CHECK: mem m$x : UInt<32>[2]
      ; CHECK: mem m$y : UInt<32>[2]

      accessor c = m[i] ; MALE
      ; CHECK: accessor c$x = m$x[i]
      ; CHECK: accessor c$y = m$y[i]
      ; CHECK: c$x := j$x
      ; CHECK: j$y := c$y
      c := j

; CHECK: Finished Lower To Ground

