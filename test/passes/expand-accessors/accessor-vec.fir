; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s

circuit top :
   module top :
      wire m : UInt<32>[2][2][2]
      wire a : UInt<32>[2][2]
      wire b : UInt<32>[2]
      wire c : UInt<32>
      wire i : UInt
      wire j : UInt
      m[0][0][0] <= UInt(1)
      m[1][0][0] <= UInt(1)
      m[0][1][0] <= UInt(1)
      m[1][1][0] <= UInt(1)
      m[0][0][1] <= UInt(1)
      m[1][0][1] <= UInt(1)
      m[0][1][1] <= UInt(1)
      m[1][1][1] <= UInt(1)
      i <= UInt(1)
      a <= m[i]
      b <= a[i]
      c <= b[i]
      j <= c

      wire x : UInt<32>[2][2]
      wire y : UInt<32>[2]
      wire z : UInt<32>
      x[0][0] <= UInt(1)
      x[1][0] <= UInt(1)
      x[0][1] <= UInt(1)
      x[1][1] <= UInt(1)
      y[0] <= UInt(1)
      y[1] <= UInt(1)
      m[i] <= x
      x[i] <= y
      y[i] <= z
      z <= j

      wire p : {n : UInt<32>[2]}
      p.n[0] <= UInt(1)
      p.n[1] <= UInt(1)
      wire q : UInt<32>
      p.n[i] <= q
      q <= j

      wire r : {m : UInt<32>}[2]
      r[0].m <= UInt(1)
      r[1].m <= UInt(1)
      wire s : { m : UInt<32>}
      s <= r[i] 
      j <= s.m

; CHECK: Done!
