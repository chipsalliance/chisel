; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
; CHECK: Primop mux requires all operands to have the same type.
; CHECK: Primop add cannot operate on non-ground types.
; CHECK: Primop mux requires argument SInt("h1") to be a UInt type.

circuit Top : 
  module Top :
    node x = mux(UInt(1),SInt(1),UInt(1))
    wire a : { q : UInt<1> }
    node y = add(a,a)
    node z = bits(SInt<10>(-1),1,2)
    node zz = mux(SInt(1),UInt(1),UInt(1))

