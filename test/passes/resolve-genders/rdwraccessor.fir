; RUN: firrtl -i %s -o %s.v -X verilog -p cg 2>&1 | tee %s.out | FileCheck %s

;CHECK: Resolve Genders
circuit top :
   module top :
      wire m : UInt<32>[2][2][2]
      m[0][0][0] <= UInt(1)
      m[1][0][0] <= UInt(1)
      m[0][1][0] <= UInt(1)
      m[1][1][0] <= UInt(1)
      m[0][0][1] <= UInt(1)
      m[1][0][1] <= UInt(1)
      m[0][1][1] <= UInt(1)
      m[1][1][1] <= UInt(1)
      wire i : UInt
      i <= UInt(1)
      rdwr accessor a = m[i] ;CHECK: accessor a = m@<g:b>[i@<g:m>]@<g:b>
      rdwr accessor b = a[i] ;CHECK: accessor b = a@<g:b>[i@<g:m>]@<g:b>
      rdwr accessor c = b[i] ;CHECK: accessor c = b@<g:b>[i@<g:m>]@<g:b>
      wire j : UInt
      j <= c
      c <= j

      rdwr accessor x = m[i] ;CHECK: accessor x = m@<g:b>[i@<g:m>]@<g:b>
      rdwr accessor y = x[i] ;CHECK: accessor y = x@<g:b>[i@<g:m>]@<g:b>
      rdwr accessor z = y[i] ;CHECK: accessor z = y@<g:b>[i@<g:m>]@<g:b>
      z <= j
      j <= z

; CHECK: Finished Resolve Genders
; CHECK: Done!
