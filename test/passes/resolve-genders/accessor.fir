; RUN: firrtl %s abcdef cg | tee %s.out | FileCheck %s

;CHECK: Resolve Genders
circuit top :
   module top :
      wire m : UInt(32)[10][10][10]
      wire i : UInt
      accessor a = m[i] ;CHECK: accessor a = m@<g:male>[i@<g:male>]@<g:male>
      accessor b = a[i] ;CHECK: accessor b = a@<g:male>[i@<g:male>]@<g:male>
      accessor c = b[i] ;CHECK: accessor c = b@<g:male>[i@<g:male>]@<g:male>
      wire j : UInt
      j := c

      accessor x = m[i] ;CHECK: accessor x = m@<g:female>[i@<g:male>]@<g:female>
      accessor y = x[i] ;CHECK: accessor y = x@<g:female>[i@<g:male>]@<g:female>
      accessor z = y[i] ;CHECK: accessor z = y@<g:female>[i@<g:male>]@<g:female>
      z := j

; CHECK: Finished Resolve Genders
