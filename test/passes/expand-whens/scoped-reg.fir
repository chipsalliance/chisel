; RUN: firrtl -i %s -o %s.flo -X flo -p c | tee %s.out | FileCheck %s
; CHECK: Expand Whens
circuit top :
   module top :
      wire p : UInt
      when p :
         reg r : UInt
         on-reset r := UInt(10)
         r := UInt(20)
; CHECK: r := Register(mux(reset, UInt(10), UInt(20)), mux(reset, UInt(1), p))
; CHECK: Finished Expand Whens

