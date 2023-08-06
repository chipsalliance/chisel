; RUN: firtool %s -split-verilog -o %t
; RUN: cat %t/ref_Top_Top.sv | FileCheck %s

; CHECK:      `define ref_Top_Top_direct_probe direct_probe
; CHECK-NEXT: `define ref_Top_Top_inner_probe inner.x_probe
; CHECK-NEXT: `define ref_Top_Top_keyword_probe always_probe

FIRRTL version 3.0.0
circuit Top:
  module Top:
    output direct_probe: Probe<UInt<1>>
    output inner_probe: Probe<UInt<2>>
    output keyword_probe: Probe<UInt<3>>

    wire direct: UInt<1>
    invalidate direct
    define direct_probe = probe(direct)

    inst inner of Inner
    define inner_probe = inner.x_probe

    wire always: UInt<3>
    invalidate always
    define keyword_probe = probe(always)

  module Inner:
    output x_probe: Probe<UInt<2>>
    wire x: UInt<2>
    invalidate x
    define x_probe = probe(x)
