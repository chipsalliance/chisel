; RUN: firrtl -i %s -o %s.flo -X flo -p cdg | tee %s.out | FileCheck %s

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
      wire connect2 : { flip data : UInt<16>, ready: UInt<1> }
      inst src of source         
      inst snk of sink           
      connect := src   
      connect2 := snk


; CHECK: Resolve Genders

; CHECK: connect@<g:f> := src@<g:m>
; CHECK: connect2@<g:f> := snk@<g:m>

; CHECK: Finished Resolve Genders


; CHECK: Lower To Ground

; CHECK: connect_data@<g:f> := src@<g:m>.data@<g:m>
; CHECK: src@<g:m>.ready@<g:f> := connect_ready@<g:m>
; CHECK: snk@<g:m>.data@<g:f> := connect2_data@<g:m>
; CHECK: connect2_ready@<g:f> := snk@<g:m>.ready@<g:m>

; CHECK: Finished Lower To Ground
