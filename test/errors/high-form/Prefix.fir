; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
; CHECK: Symbol x$z$y is a prefix.

circuit Top :
  module Top :
    wire x : UInt<2>
    wire x$z$y : UInt<1>


