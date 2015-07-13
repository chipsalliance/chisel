; RUN: firrtl -i %s -o %s.flo -X flo -p c | tee %s.out | FileCheck %s

; CHECK: Expand Indexed Connects
circuit top :
   module top :
      wire i : UInt
      i := UInt(1)
      wire j : UInt
      j := UInt(1)

      wire a : { x : UInt<32>, flip y : UInt<32> }[2]
      a[0].x := UInt(1)
      a[0].y := UInt(1)
      a[1].x := UInt(1)
      a[1].y := UInt(1)
      ; CHECK: wire a_0_x : UInt<32>
      ; CHECK: wire a_0_y : UInt<32>
      ; CHECK: wire a_1_x : UInt<32>
      ; CHECK: wire a_1_y : UInt<32>


      infer accessor b = a[i]
      ; CHECK: wire b_x : UInt<32>
      ; CHECK: wire b_y : UInt<32>
      ; CHECK: b_x := a_0_x
      ; CHECK: node i!0 = i
      ; CHECK: when eq(i!0, UInt(1)) : b_x := a_1_x      
      ; CHECK: node i!1 = i
      ; CHECK: when eq(i!1, UInt(0)) : a_0_y := b_y      
      ; CHECK: when eq(i!1, UInt(1)) : a_1_y := b_y      
      j := b.x
      b.y := UInt(1)
      
; CHECK: Finished Expand Indexed Connects

