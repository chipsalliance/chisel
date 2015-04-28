; RUN: firrtl -i %s -o %s.flo -x abcdefg -p c | tee %s.out | FileCheck %s

;CHECK: Expand Accessors
circuit top :
   module top :
      wire m : UInt<32>[10][10][10]
      wire i : UInt
      accessor a = m[i] ;CHECK: a := (m[0] m[1] m[2] m[3] m[4] m[5] m[6] m[7] m[8] m[9])[i]
      accessor b = a[i] ;CHECK: b := (a[0] a[1] a[2] a[3] a[4] a[5] a[6] a[7] a[8] a[9])[i]
      accessor c = b[i] ;CHECK: c := (b[0] b[1] b[2] b[3] b[4] b[5] b[6] b[7] b[8] b[9])[i]
      wire j : UInt
      j := c

      accessor x = m[i] ;CHECK: (m[0] m[1] m[2] m[3] m[4] m[5] m[6] m[7] m[8] m[9])[i] := x
      accessor y = x[i] ;CHECK: (x[0] x[1] x[2] x[3] x[4] x[5] x[6] x[7] x[8] x[9])[i] := y
      accessor z = y[i] ;CHECK: (y[0] y[1] y[2] y[3] y[4] y[5] y[6] y[7] y[8] y[9])[i] := z
      z := j

      wire p : {n : UInt<32>[10]}
      accessor q = p.n[i] ;CHECK: (p.n[0] p.n[1] p.n[2] p.n[3] p.n[4] p.n[5] p.n[6] p.n[7] p.n[8] p.n[9])[i] := q
      q := j

      wire r : {m : UInt<32>}[10]
      accessor s = r[i] ;CHECK: s := (r[0] r[1] r[2] r[3] r[4] r[5] r[6] r[7] r[8] r[9])[i]
      j := s.m

; CHECK: Finished Expand Accessors
