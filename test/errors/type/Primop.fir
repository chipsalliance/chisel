; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
; CHECK: Must mux between equivalent types.
; CHECK: Primop add cannot operate on non-ground types.
; CHECK: A mux condition must be of type UInt.

circuit Top : 
  module Top :
    node x = mux(UInt(1),SInt(1),UInt(1))
    wire a : { q : UInt<1> }
    node y = add(a,a)
    node z = bits(SInt<10>(-1),1,2)
    node zz = mux(SInt(1),UInt(1),UInt(1))

