; RUN: firrtl -i %s -o %s.flo -x X -p c | tee %s.out | FileCheck %s
; CHECK: FIRRTL Parsing Error: Expected a reference expression here.

circuit Top :
  module Top :
    wire x : UInt<1>
    inst t of add(UInt(1),UInt(1))

