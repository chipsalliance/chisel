; RUN: firrtl -i %s -o %s.flo -X flo -p c | tee %s.out | FileCheck %s

; CHECK: Expand Indexed Connects
circuit top :
   module top :
      wire i : UInt
      wire j : UInt

      wire a : { x : UInt<32>, flip y : UInt<32> }[2]
      ; CHECK: wire a$0$x : UInt<32>
      ; CHECK: wire a$0$y : UInt<32>
      ; CHECK: wire a$1$x : UInt<32>
      ; CHECK: wire a$1$y : UInt<32>

      accessor b = a[i]
      ; CHECK: wire b$x : UInt<32>
      ; CHECK: wire b$y : UInt<32>
      ; CHECK: b$x := a$0$x
      ; CHECK: node b$x#0 = i
      ; CHECK: when eq(b$x#0, UInt(1)) :
      ; CHECK:    b$x := a$1$x      
      ; CHECK: node b$y#0 = i
      ; CHECK: when eq(b$y#0, UInt(0)) :
      ; CHECK:    a$0$y := b$y      
      ; CHECK: when eq(b$y#0, UInt(1)) :
      ; CHECK:    a$1$y := b$y      
      j := b.x
      
; CHECK: Finished Expand Indexed Connects

