; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
; CHECK: Primop mux requires 3 expression arguments.
; CHECK: Primop add requires 2 expression arguments.
; CHECK: Primop bits requires 2 integer arguments.

circuit Top : 
  module Top : 
    node x = mux(UInt(1),UInt(1))
    node y = add(SInt(1),UInt(1),UInt(1))
    node z = bits(UInt(1),1,2,3)

