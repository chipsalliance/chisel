; RUN: firrtl -i %s -o %s.v -X verilog -p cg 2>&1 | tee %s.out | FileCheck %s

;CHECK: Resolve Genders
circuit top :
   module source :
      output data : UInt<16>
      input ready : UInt<1>
      data := UInt(16)
   module sink :
      input data : UInt<16>
      output ready : UInt<1>
   module top: 
      wire connect : { data : UInt<16>, flip ready: UInt<1> }
      inst src of source         ;CHECK: inst src of source@<g:m>
      inst snk of sink           ;CHECK: inst snk of sink@<g:m>
      connect.data := src.data   ;CHECK: connect@<g:f>.data@<g:f> := src@<g:m>.data@<g:m>
      src.ready := connect.ready ;CHECK: src@<g:m>.ready@<g:f> := connect@<g:f>.ready@<g:m>
      snk.data := connect.data   ;CHECK: snk@<g:m>.data@<g:f> := connect@<g:m>.data@<g:m>
      connect.ready := snk.ready ;CHECK: connect@<g:m>.ready@<g:f> := snk@<g:m>.ready@<g:m>

; CHECK: Finished Resolve Genders
