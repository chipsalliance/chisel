; RUN: firrtl -i %s -o %s.flo -X flo -p c | tee %s.out | FileCheck %s
circuit top :
   module top :
      wire p : UInt
      p := UInt(1)
      when p :
         reg r : UInt
         r := UInt(20)

; CHECK: Expand Whens

; CHECK: circuit top :
; CHECK:   module top :
; CHECK:     wire p : UInt
; CHECK:     reg r : UInt
; CHECK:     p := UInt(1)
; CHECK:     r := UInt(20)

; CHECK: Finished Expand Whens

