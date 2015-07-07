; RUN: firrtl -i %s -o %s.flo -X flo -p cd | tee %s.out | FileCheck %s

; CHECK: Lower To Ground
circuit top :
   module top :
      wire i : UInt<2>
      wire j : UInt<32>

      wire a : UInt<32>[4]
      ; CHECK: wire a$0 : UInt<32>
      ; CHECK: wire a$1 : UInt<32>
      ; CHECK: wire a$2 : UInt<32>
      ; CHECK: wire a$3 : UInt<32>

      infer accessor b = a[i]
      ; CHECK: wire b : UInt<32>
      ; CHECK: b := (a$0 a$1 a$2 a$3)[i]
      j := b

      infer accessor c = a[i]
      ; CHECK: wire c : UInt<32>
      ; CHECK: (a$0 a$1 a$2 a$3)[i] := c
      c := j
      
      cmem p : UInt<32>[4]
      infer accessor t = p[i]
      ; CHECK: accessor t = p[i]
      j := t
      infer accessor r = p[i]
      ; CHECK: accessor r = p[i]
      r := j

; CHECK: Finished Lower To Ground
