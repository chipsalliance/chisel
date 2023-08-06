; RUN: firtool --ir-hw %s | FileCheck %s

circuit Foo:
  intmodule HasBeenReset:
    input clock: Clock
    input reset: Reset
    output out: UInt<1>
    intrinsic = circt_has_been_reset

  module Foo:
    input clock: Clock
    input reset1: UInt<1>
    input reset2: AsyncReset
    output hbr: UInt<1>[2]

    wire resetWire1 : Reset
    wire resetWire2 : Reset

    resetWire1 <= reset1
    resetWire2 <= reset2

    inst int1 of HasBeenReset
    int1.clock <= clock
    int1.reset <= resetWire1
    hbr[0] <= int1.out

    inst int2 of HasBeenReset
    int2.clock <= clock
    int2.reset <= resetWire2
    hbr[1] <= int2.out

; CHECK-LABEL: hw.module @Foo
; CHECK-NEXT: verif.has_been_reset %clock, sync %reset1
; CHECK-NEXT: verif.has_been_reset %clock, async %reset2
; CHECK-NEXT: hw.output
