; RUN: firrtl -i %s -o %s.v -X verilog -p cg 2>&1 | tee %s.out | FileCheck %s

;CHECK: Inline Indexers
;CHECK: Done!

circuit DecoupledAdderTests : 
  module DecoupledAdderTests : 
    input clock : Clock
    input reset : UInt<1>
    input T_31 : UInt<1>
    input T_68 : UInt<1>
    output out : UInt
    output io : {}
    wire T_43 : {flip ready : UInt<1>}[1]
    T_43[0].ready := UInt(0)
    infer accessor T_69 = T_43[T_31]
    node T_78 = and(T_68, T_69.ready)
    out := T_78
