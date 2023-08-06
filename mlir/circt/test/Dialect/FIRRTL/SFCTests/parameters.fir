; RUN: firtool -split-input-file -verilog %s | FileCheck %s

; Test that a negative parameter prints out as a 32-bit parameter.  It is fine
; to change this test to print as "-1" in the output Verilog, but not as a
; non-32-bit "-1" like "0xF".
circuit NegativeParameter:
  extmodule Foo:
    output a: UInt<1>
    parameter x = -1
  module NegativeParameter:
    output a: UInt<1>

    inst foo of Foo
    a <= foo.a

; CHECK-LABEL: module NegativeParameter
; CHECK:         Foo #(
; CHECK-NEXT:      .x(-1)
