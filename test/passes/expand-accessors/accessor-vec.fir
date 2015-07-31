; RUN: firrtl -i %s -o %s.flo -X flo -p c 2>&1 | tee %s.out | FileCheck %s

;CHECK: Expand Accessors
circuit top :
   module top :
      wire m : UInt<32>[2][2][2]
      m[0][0][0] := UInt(1)
      m[1][0][0] := UInt(1)
      m[0][1][0] := UInt(1)
      m[1][1][0] := UInt(1)
      m[0][0][1] := UInt(1)
      m[1][0][1] := UInt(1)
      m[0][1][1] := UInt(1)
      m[1][1][1] := UInt(1)
      wire i : UInt
      i := UInt(1)
      infer accessor a = m[i] ;CHECK: a := (m[0] m[1])[i]
      infer accessor b = a[i] ;CHECK: b := (a[0] a[1])[i]
      infer accessor c = b[i] ;CHECK: c := (b[0] b[1])[i]
      wire j : UInt
      j := c

      infer accessor x = m[i] ;CHECK: (m[0] m[1])[i] := x
      x[0][0] := UInt(1)
      x[1][0] := UInt(1)
      x[0][1] := UInt(1)
      x[1][1] := UInt(1)
      infer accessor y = x[i] ;CHECK: (x[0] x[1])[i] := y
      infer accessor z = y[i] ;CHECK: (y[0] y[1])[i] := z
      y[0] := UInt(1)
      y[1] := UInt(1)
      z := j

      wire p : {n : UInt<32>[2]}
      p.n[0] := UInt(1)
      p.n[1] := UInt(1)
      infer accessor q = p.n[i] ;CHECK: (p.n[0] p.n[1])[i] := q
      q := j

      wire r : {m : UInt<32>}[2]
      r[0].m := UInt(1)
      r[1].m := UInt(1)
      infer accessor s = r[i] ;CHECK: s := (r[0] r[1])[i]
      j := s.m

; CHECK: Finished Expand Accessors
