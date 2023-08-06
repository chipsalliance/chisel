; RUN: firtool %s | FileCheck %s --check-prefixes=CHECK,CHECK-DEFAULT
; RUN: firtool %s --ckg-name=MyClockGate --ckg-input=CI --ckg-output=CO --ckg-enable=E --ckg-test-enable=TE | FileCheck %s --check-prefixes=CHECK,CHECK-CUSTOM

circuit Foo:
  intmodule ClockGate:
    input in: Clock
    input en: UInt<1>
    output out: Clock
    intrinsic = circt_clock_gate

  module Foo:
    input clk: Clock
    input enable: UInt<1>
    output gated_clk: Clock

    inst ckg of ClockGate
    ckg.in <= clk
    ckg.en <= enable
    gated_clk <= ckg.out

; CHECK-LABEL: module Foo(

; CHECK-DEFAULT: EICG_wrapper ckg (
; CHECK-DEFAULT: .in (clk)
; CHECK-DEFAULT: .en (enable)
; CHECK-DEFAULT: .test_en (1'h0)
; CHECK-DEFAULT: .out (gated_clk)

; CHECK-CUSTOM: MyClockGate ckg (
; CHECK-CUSTOM: .CI (clk)
; CHECK-CUSTOM: .E (enable)
; CHECK-CUSTOM: .TE (1'h0)
; CHECK-CUSTOM: .CO (gated_clk)
