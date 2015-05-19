; RUN: firrtl -i %s -o %s.flo -X flo -p c | tee %s.out | FileCheck %s

;CHECK: Expand Accessors
circuit top :
   module top :
      wire m : UInt<32>[2][2][2]
      wire i : UInt
      i := UInt(1)
      accessor a = m[i] ;CHECK: a := (m[0] m[1])[i]
      accessor b = a[i] ;CHECK: b := (a[0] a[1])[i]
      accessor c = b[i] ;CHECK: c := (b[0] b[1])[i]
      wire j : UInt
      j := c

      accessor x = m[i] ;CHECK: (m[0] m[1])[i] := x
      accessor y = x[i] ;CHECK: (x[0] x[1])[i] := y
      accessor z = y[i] ;CHECK: (y[0] y[1])[i] := z
      z := j

      wire p : {n : UInt<32>[2]}
      accessor q = p.n[i] ;CHECK: (p.n[0] p.n[1])[i] := q
      q := j

      wire r : {m : UInt<32>}[2]
      accessor s = r[i] ;CHECK: s := (r[0] r[1])[i]
      j := s.m

; CHECK: Finished Expand Accessors
