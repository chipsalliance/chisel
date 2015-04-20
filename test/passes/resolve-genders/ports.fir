; RUN: firrtl -i %s -o %s.flo -x abcdef -p cg | tee %s.out | FileCheck %s

;CHECK: Resolve Genders
circuit top :
   module source :
      output data : UInt(16)
      input ready : UInt(1)
      data := UInt(16)
   module sink :
      input data : UInt(16)
      output ready : UInt(1)
   module top: 
      wire connect : { data : UInt(16), flip ready: UInt(1) }
      inst src of source         ;CHECK: inst src of source@<g:female>
      inst snk of sink           ;CHECK: inst snk of sink@<g:female>
      connect.data := src.data   ;CHECK: connect@<g:female>.data@<g:female> := src@<g:female>.data@<g:male>
      src.ready := connect.ready ;CHECK: src@<g:female>.ready@<g:female> := connect@<g:female>.ready@<g:male>
      snk.data := connect.data   ;CHECK: snk@<g:female>.data@<g:female> := connect@<g:male>.data@<g:male>
      connect.ready := snk.ready ;CHECK: connect@<g:male>.ready@<g:female> := snk@<g:female>.ready@<g:male>

; CHECK: Finished Resolve Genders
