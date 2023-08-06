; RUN: rm -rf %t && mkdir %t && firtool --split-verilog --annotation-file %s.anno.json --annotation-file %s.extract.anno.json -o %t %s && cd %t && verilator --sv --lint-only Top.sv bindings.sv
; RUN: firtool --verilog --annotation-file %s.anno.json %s > %t.one-file.sv && verilator --sv --lint-only %t.one-file.sv
; REQUIRES: verilator

circuit Top :
  module Submodule :
    input clock : Clock
    input reset : Reset
    input in : { uint : UInt<1>, vec : UInt<1>[2], vecOfBundle : { uint : UInt<4>, sint : SInt<2>}[2], otherOther : { other : { uint : UInt<4>, sint : SInt<2>}}}
    output out : { uint : UInt<1>, vec : UInt<1>[2], vecOfBundle : { uint : UInt<4>, sint : SInt<2>}[2], otherOther : { other : { uint : UInt<4>, sint : SInt<2>}}}

    wire w : { uint : UInt<1>, vec : UInt<1>[2], vecOfBundle : { uint : UInt<4>, sint : SInt<2>}[2], otherOther : { other : { uint : UInt<4>, sint : SInt<2>}}}
    w.otherOther.other.sint <= in.otherOther.other.sint
    w.otherOther.other.uint <= in.otherOther.other.uint
    w.vecOfBundle[0].sint <= in.vecOfBundle[0].sint
    w.vecOfBundle[0].uint <= in.vecOfBundle[0].uint
    w.vecOfBundle[1].sint <= in.vecOfBundle[1].sint
    w.vecOfBundle[1].uint <= in.vecOfBundle[1].uint
    w.vec[0] <= in.vec[0]
    w.vec[1] <= in.vec[1]
    w.uint <= in.uint
    out.otherOther.other.sint <= w.otherOther.other.sint
    out.otherOther.other.uint <= w.otherOther.other.uint
    out.vecOfBundle[0].sint <= w.vecOfBundle[0].sint
    out.vecOfBundle[0].uint <= w.vecOfBundle[0].uint
    out.vecOfBundle[1].sint <= w.vecOfBundle[1].sint
    out.vecOfBundle[1].uint <= w.vecOfBundle[1].uint
    out.vec[0] <= w.vec[0]
    out.vec[1] <= w.vec[1]
    out.uint <= w.uint

  module MyView_companion :
    output io : { }

    wire _WIRE : UInt<1>
    _WIRE <= UInt<1>("h0")

  module DUT :
    input clock : Clock
    input reset : Reset
    input in : { uint : UInt<1>, vec : UInt<1>[2], vecOfBundle : { uint : UInt<4>, sint : SInt<2>}[2], otherOther : { other : { uint : UInt<4>, sint : SInt<2>}}}
    output out : { uint : UInt<1>, vec : UInt<1>[2], vecOfBundle : { uint : UInt<4>, sint : SInt<2>}[2], otherOther : { other : { uint : UInt<4>, sint : SInt<2>}}}

    wire w : { uint : UInt<1>, vec : UInt<1>[2], vecOfBundle : { uint : UInt<4>, sint : SInt<2>}[2], otherOther : { other : { uint : UInt<4>, sint : SInt<2>}}}
    inst submodule of Submodule
    submodule.clock <= clock
    submodule.reset <= reset
    w.otherOther.other.sint <= in.otherOther.other.sint
    w.otherOther.other.uint <= in.otherOther.other.uint
    w.vecOfBundle[0].sint <= in.vecOfBundle[0].sint
    w.vecOfBundle[0].uint <= in.vecOfBundle[0].uint
    w.vecOfBundle[1].sint <= in.vecOfBundle[1].sint
    w.vecOfBundle[1].uint <= in.vecOfBundle[1].uint
    w.vec[0] <= in.vec[0]
    w.vec[1] <= in.vec[1]
    w.uint <= in.uint
    submodule.in.otherOther.other.sint <= w.otherOther.other.sint
    submodule.in.otherOther.other.uint <= w.otherOther.other.uint
    submodule.in.vecOfBundle[0].sint <= w.vecOfBundle[0].sint
    submodule.in.vecOfBundle[0].uint <= w.vecOfBundle[0].uint
    submodule.in.vecOfBundle[1].sint <= w.vecOfBundle[1].sint
    submodule.in.vecOfBundle[1].uint <= w.vecOfBundle[1].uint
    submodule.in.vec[0] <= w.vec[0]
    submodule.in.vec[1] <= w.vec[1]
    submodule.in.uint <= w.uint
    out.otherOther.other.sint <= submodule.out.otherOther.other.sint
    out.otherOther.other.uint <= submodule.out.otherOther.other.uint
    out.vecOfBundle[0].sint <= submodule.out.vecOfBundle[0].sint
    out.vecOfBundle[0].uint <= submodule.out.vecOfBundle[0].uint
    out.vecOfBundle[1].sint <= submodule.out.vecOfBundle[1].sint
    out.vecOfBundle[1].uint <= submodule.out.vecOfBundle[1].uint
    out.vec[0] <= submodule.out.vec[0]
    out.vec[1] <= submodule.out.vec[1]
    out.uint <= submodule.out.uint
    inst MyView_companion of MyView_companion

  module Top :
    input clock : Clock
    input reset : UInt<1>
    input in : { uint : UInt<1>, vec : UInt<1>[2], vecOfBundle : { uint : UInt<4>, sint : SInt<2>}[2], otherOther : { other : { uint : UInt<4>, sint : SInt<2>}}}
    output out : { uint : UInt<1>, vec : UInt<1>[2], vecOfBundle : { uint : UInt<4>, sint : SInt<2>}[2], otherOther : { other : { uint : UInt<4>, sint : SInt<2>}}}

    inst dut of DUT
    dut.clock <= clock
    dut.reset <= reset
    dut.in.otherOther.other.sint <= in.otherOther.other.sint
    dut.in.otherOther.other.uint <= in.otherOther.other.uint
    dut.in.vecOfBundle[0].sint <= in.vecOfBundle[0].sint
    dut.in.vecOfBundle[0].uint <= in.vecOfBundle[0].uint
    dut.in.vecOfBundle[1].sint <= in.vecOfBundle[1].sint
    dut.in.vecOfBundle[1].uint <= in.vecOfBundle[1].uint
    dut.in.vec[0] <= in.vec[0]
    dut.in.vec[1] <= in.vec[1]
    dut.in.uint <= in.uint
    out.otherOther.other.sint <= dut.out.otherOther.other.sint
    out.otherOther.other.uint <= dut.out.otherOther.other.uint
    out.vecOfBundle[0].sint <= dut.out.vecOfBundle[0].sint
    out.vecOfBundle[0].uint <= dut.out.vecOfBundle[0].uint
    out.vecOfBundle[1].sint <= dut.out.vecOfBundle[1].sint
    out.vecOfBundle[1].uint <= dut.out.vecOfBundle[1].uint
    out.vec[0] <= dut.out.vec[0]
    out.vec[1] <= dut.out.vec[1]
    out.uint <= dut.out.uint
