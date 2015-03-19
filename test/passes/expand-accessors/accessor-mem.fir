; RUN: firrtl %s abcdefg c | tee %s.out | FileCheck %s

;CHECK: Expand Accessors
circuit top :
   module top :
      mem m : UInt(32)[10][10][10]
      wire i : UInt
      accessor a = m[i] ;CHECK: accessor a = m[i]
      accessor b = a[i] ;CHECK: b := (a.0 a.1 a.2 a.3 a.4 a.5 a.6 a.7 a.8 a.9)[i]
      accessor c = b[i] ;CHECK: c := (b.0 b.1 b.2 b.3 b.4 b.5 b.6 b.7 b.8 b.9)[i]
      wire j : UInt
      j := c

      accessor x = m[i] ;CHECK: accessor x = m[i]
      accessor y = x[i] ;CHECK: (x.0 x.1 x.2 x.3 x.4 x.5 x.6 x.7 x.8 x.9)[i] := y
      accessor z = y[i] ;CHECK: (y.0 y.1 y.2 y.3 y.4 y.5 y.6 y.7 y.8 y.9)[i] := z
      z := j

; CHECK: Finished Expand Accessors
