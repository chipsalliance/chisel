; RUN: firrtl -i %s -o %s.flo -x X -p c | tee %s.out | FileCheck %s

;CHECK: Expand Accessors
circuit top :
   module top :
      mem m : UInt<32>[2][2][2]
      wire i : UInt<4>
      accessor a = m[i] ;CHECK: accessor a = m[i]
      accessor b = a[i] ;CHECK: b := (a[0] a[1])[i]
      accessor c = b[i] ;CHECK: c := (b[0] b[1])[i]
      wire j : UInt
      j := c

      accessor x = m[i] ;CHECK: accessor x = m[i]
      accessor y = x[i] ;CHECK: (x[0] x[1])[i] := y
      accessor z = y[i] ;CHECK: (y[0] y[1])[i] := z
      z := j

; CHECK: Finished Expand Accessors
