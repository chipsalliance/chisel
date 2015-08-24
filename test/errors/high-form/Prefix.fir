; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
; CHECK: Reference x$y and x share a prefix.
; XFAIL: *

circuit Top :
  module Top :
    wire x : UInt<1>
    wire x$y : UInt<2>


