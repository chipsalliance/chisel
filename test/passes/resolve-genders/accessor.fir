; RUN: firrtl -i %s -o %s.flo -X flo -p cg | tee %s.out | FileCheck %s

;CHECK: Resolve Genders
circuit top :
   module top :
      wire m : UInt<32>[5][5][5]
      wire i : UInt
      accessor a = m[i] ;CHECK: accessor a = m@<g:m>[i@<g:m>]@<g:m>
      accessor b = a[i] ;CHECK: accessor b = a@<g:m>[i@<g:m>]@<g:m>
      accessor c = b[i] ;CHECK: accessor c = b@<g:m>[i@<g:m>]@<g:m>
      wire j : UInt
      j := c

      accessor x = m[i] ;CHECK: accessor x = m@<g:f>[i@<g:m>]@<g:f>
      accessor y = x[i] ;CHECK: accessor y = x@<g:f>[i@<g:m>]@<g:f>
      accessor z = y[i] ;CHECK: accessor z = y@<g:f>[i@<g:m>]@<g:f>
      z := j

; CHECK: Finished Resolve Genders
