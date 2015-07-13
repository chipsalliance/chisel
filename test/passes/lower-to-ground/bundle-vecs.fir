; RUN: firrtl -i %s -o %s.flo -X flo -p c | tee %s.out | FileCheck %s

; CHECK: Lower To Ground
circuit top :
   module top :
      wire i : UInt
      wire j : { x : UInt<32>, flip y : UInt<32> }

      wire a : { x : UInt<32>, flip y : UInt<32> }[2]
      ; CHECK: wire a_0_x : UInt<32>
      ; CHECK: wire a_0_y : UInt<32>
      ; CHECK: wire a_1_x : UInt<32>
      ; CHECK: wire a_1_y : UInt<32>

      infer accessor b = a[i]
      ; CHECK: wire b_x : UInt<32>
      ; CHECK: wire b_y : UInt<32>
      ; CHECK: b_x := (a_0_x a_1_x)[i]
      ; CHECK: (a_0_y a_1_y)[i] := b_y
      j := b
      
      infer accessor c = a[i]
      ; CHECK: wire c_x : UInt<32>
      ; CHECK: wire c_y : UInt<32>
      ; CHECK: (a_0_x a_1_x)[i] := c_x
      ; CHECK: c_y := (a_0_y a_1_y)[i]
      c := j


; CHECK: Finished Lower To Ground

