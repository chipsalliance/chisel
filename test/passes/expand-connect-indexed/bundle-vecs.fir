; RUN: firrtl -i %s -o %s.flo -X flo -p c | tee %s.out | FileCheck %s

; CHECK: Expand Indexed Connects
circuit top :
   module top :
      wire i : UInt
      wire j : UInt

      wire a : { x : UInt<32>, flip y : UInt<32> }[2]
      ; CHECK: wire a_0_x : UInt<32>
      ; CHECK: wire a_0_y : UInt<32>
      ; CHECK: wire a_1_x : UInt<32>
      ; CHECK: wire a_1_y : UInt<32>

      accessor b = a[i]
      ; CHECK: wire b_x : UInt<32>
      ; CHECK: wire b_y : UInt<32>
      ; CHECK: b_x := a_0_x
      ; CHECK: node b_x__0 = i
      ; CHECK: when eq(b_x__0, UInt(1)) :
      ; CHECK:    b_x := a_1_x      
      ; CHECK: node b_y__0 = i
      ; CHECK: when eq(b_y__0, UInt(0)) :
      ; CHECK:    a_0_y := b_y      
      ; CHECK: when eq(b_y__0, UInt(1)) :
      ; CHECK:    a_1_y := b_y      
      j := b.x
      
; CHECK: Finished Expand Indexed Connects

