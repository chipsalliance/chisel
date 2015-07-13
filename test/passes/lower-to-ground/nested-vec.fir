; RUN: firrtl -i %s -o %s.flo -X flo -p c | tee %s.out | FileCheck %s

; CHECK: Lower To Ground
circuit top :
   module top :
      input clk : Clock
      wire i : UInt
      wire j : { x : UInt<32>, flip y : UInt<32> }
      wire k : { x : UInt<32>, y : UInt<32> }

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

      cmem m : { x : UInt<32>, y : UInt<32> }[2],clk
      ; CHECK: cmem m_x : UInt<32>[2]
      ; CHECK: cmem m_y : UInt<32>[2]

      infer accessor c = m[i] ; MALE
      ; CHECK: accessor c_x = m_x[i]
      ; CHECK: accessor c_y = m_y[i]

      c := k
      ; CHECK: c_x := k_x
      ; CHECK: c_y := k_y

; CHECK: Finished Lower To Ground

