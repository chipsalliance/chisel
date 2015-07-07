; RUN: firrtl -i %s -o %s.flo -X flo -p cg | tee %s.out | FileCheck %s

;CHECK: Resolve Genders
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
      infer accessor a = m[i] ;CHECK: accessor a = m@<g:m>[i@<g:m>]@<g:m>
      infer accessor b = a[i] ;CHECK: accessor b = a@<g:m>[i@<g:m>]@<g:m>
      infer accessor c = b[i] ;CHECK: accessor c = b@<g:m>[i@<g:m>]@<g:m>
      wire j : UInt
      j := c

      infer accessor x = m[i] ;CHECK: accessor x = m@<g:f>[i@<g:m>]@<g:f>
      x[0][0] := UInt(1)
      x[1][0] := UInt(1)
      x[0][1] := UInt(1)
      x[1][1] := UInt(1)
      infer accessor y = x[i] ;CHECK: accessor y = x@<g:f>[i@<g:m>]@<g:f>
      y[0] := UInt(1)
      y[1] := UInt(1)
      infer accessor z = y[i] ;CHECK: accessor z = y@<g:f>[i@<g:m>]@<g:f>
      z := j

; CHECK: Finished Resolve Genders
