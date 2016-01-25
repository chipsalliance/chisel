; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
; CHECK: [module Top]  Must mux between passive types.
; CHECK: [module Top]  A mux condition must be of type UInt.

circuit Top : 
  module Top : 
    input c : SInt<1>
    input x : {x : UInt, flip y : UInt}
    input y : {x : UInt, flip y : UInt}
    output z : {x : UInt, flip y : UInt}
    z <= mux(c,x,y)

