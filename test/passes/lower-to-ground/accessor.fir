; RUN: firrtl -i %s -o %s.flo -X flo -p cd | tee %s.out | FileCheck %s

; CHECK: Lower To Ground
circuit top :
   module top :
      wire i : UInt<2>
      wire j : UInt<32>

      wire a : UInt<32>[4]
      ; CHECK: wire a_0 : UInt<32>
      ; CHECK: wire a_1 : UInt<32>
      ; CHECK: wire a_2 : UInt<32>
      ; CHECK: wire a_3 : UInt<32>

      accessor b = a[i]
      ; CHECK: wire b : UInt<32>
      ; CHECK: b := (a_0 a_1 a_2 a_3)[i]
      j := b

      accessor c = a[i]
      ; CHECK: wire c : UInt<32>
      ; CHECK: (a_0 a_1 a_2 a_3)[i] := c
      c := j
      
      mem p : UInt<32>[4]
      accessor t = p[i]
      ; CHECK: accessor t = p[i]
      j := t
      accessor r = p[i]
      ; CHECK: accessor r = p[i]
      r := j

; CHECK: Finished Lower To Ground
