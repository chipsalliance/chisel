; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
; CHECK: Expression out is used as a source but can only be used as a sink.

circuit BTB :
  module BTB : 
    output out : {x : UInt<1>, flip y : UInt<1>}

    wire w : {x : UInt<1>, flip y : UInt<1>}
    w.x <= UInt(1)
    w.y <= UInt(1)
    out.x <= UInt(1)
    w <- out
