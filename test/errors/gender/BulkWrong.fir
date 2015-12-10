; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
; CHECK: Expression in is used as a sink but can only be used as a source.
; CHECK: Expression out.y is used as a sink but can only be used as a source.
; CHECK: Expression in.y.z is used as a sink but can only be used as a source.
; CHECK: Expression in.y.z is used as a sink but can only be used as a source.

circuit BTB :
  module BTB : 
    input in : {x : UInt<1>, flip y : {flip z : UInt<1>}}
    output out : {x : UInt<1>, flip y : {flip z : UInt<1>}}

    in <- out
    out.y <- in.y
    out.y.z <- in.y.z
    
    wire w : {x : UInt<1>, flip y : {flip z : UInt<1>}}
    w <- in
    in.y <- w.y
    in.y.z <- w.y.z

    w.x <= addw(in.x,in.y.z)

    out <- in
    in.y <- out.y
    in.y.z <- out.y.z

