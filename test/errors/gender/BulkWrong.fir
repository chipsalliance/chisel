; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
; CHECK: Expression req is used as a sink but can only be used as a source.

circuit BTB :
  module BTB : 
    input clk : Clock
    input reset : UInt<1>
    input req : {valid : UInt<1>, bits : {addr : UInt<39>}}

    output r : { x : UInt<1>, flip y : UInt<1>}

    wire x : {valid : UInt<1>, bits : {addr : UInt<39>}}
    req <> x

    wire z : {x : UInt<1>, flip y : UInt<1> }
    x.valid := r
