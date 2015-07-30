; RUN: firrtl -i %s -o %s.flo -X flo -p c | tee %s.out | FileCheck %s
; CHECK: Reference x$y and x share a prefix.
; XFAIL: *

circuit Top :
  module Top :
    wire x : UInt<1>
    wire x$y : UInt<2>


