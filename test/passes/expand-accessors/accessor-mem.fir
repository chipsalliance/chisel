; RUN: firrtl -i %s -o %s.flo -X flo -p c | tee %s.out | FileCheck %s

;CHECK: Expand Accessors
circuit top :
   module top :
      cmem m : UInt<32>[2][2][2]
      wire i : UInt<4>
      i := UInt(1)
      infer accessor a = m[i] ;CHECK: read accessor a = m[i]
      infer accessor b = a[i] ;CHECK: b := (a[0] a[1])[i]
      infer accessor c = b[i] ;CHECK: c := (b[0] b[1])[i]
      wire j : UInt
      j := c

      infer accessor x = m[i] ;CHECK: write accessor x = m[i]
      infer accessor y = x[i] ;CHECK: (x[0] x[1])[i] := y
      y[0] := UInt(1)
      y[1] := UInt(1)
      infer accessor z = y[i] ;CHECK: (y[0] y[1])[i] := z
      z := j

; CHECK: Finished Expand Accessors
