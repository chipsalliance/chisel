; RUN: firrtl -i %s -o %s.out -X firrtl 2>&1 | tee %s.out | FileCheck %s
: XFAIL: *

circuit top :
   module top :
      output a : UInt
      output b : SInt
      output c : SInt
      a <= UInt(8)
      b <= SInt(8)
      c <= SInt(-8)

; CHECK: circuit top :
; CHECK:   module top :
; CHECK:      output a : UInt
; CHECK:      output b : SInt
; CHECK:      output c : SInt
; CHECK:      a <= UInt<4>("h8")
; CHECK:      b <= SInt<5>("h08")
; CHECK:      c <= SInt<4>("h8")
