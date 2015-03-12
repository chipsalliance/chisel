; RUN: firrtl %s abcdefg c | tee %s.out | FileCheck %s

;CHECK: Expand Accessors
circuit top :
   module top :
      mem m : UInt(32)[10][10][10]
      wire i : UInt
      accessor a = m[i] ;CHECK: a := ReadPort(m, i, UInt(1))
      accessor b = a[i] ;CHECK: b := (a.9 a.8 a.7 a.6 a.5 a.4 a.3 a.2 a.1 a.0)[i]
      accessor c = b[i] ;CHECK: c := (b.9 b.8 b.7 b.6 b.5 b.4 b.3 b.2 b.1 b.0)[i]
      wire j : UInt
      j := c

      accessor x = m[i] ;CHECK: WritePort(m, i, UInt(1)) := x
      accessor y = x[i] ;CHECK: (x.9 x.8 x.7 x.6 x.5 x.4 x.3 x.2 x.1 x.0)[i] := y
      accessor z = y[i] ;CHECK: (y.9 y.8 y.7 y.6 y.5 y.4 y.3 y.2 y.1 y.0)[i] := z
      z := j

; CHECK: Finished Expand Accessors
