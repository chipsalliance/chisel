; RUN: firtool %s -extract-test-code | FileCheck %s
; RUN: firtool %s -extract-test-code -etc-disable-instance-extraction | FileCheck %s --check-prefix=DISABLEINSTANCE
; RUN: firtool %s -extract-test-code -etc-disable-module-inlining | FileCheck %s --check-prefix=DISABLEMODULE

circuit Top:
  module Foo:
    input a : UInt<1>
    output b : UInt<1>
    b <= a

  ; Ensure foo is extracted by default.
  ; CHECK-LABEL: module InstanceExtracted_assert(
  ; CHECK: Foo foo

  ; Ensure foo is not extracted when disabled.
  ; DISABLEINSTANCE-LABEL: module InstanceExtracted(
  ; DISABLEINSTANCE: Foo foo

  module InstanceExtracted:
    input clock : Clock
    input cond : UInt<1>
    output out : UInt<1>

    wire b : UInt<1>
    inst foo of Foo
    foo.a <= cond
    b <= foo.b

    assert(clock, cond, b, "Some assertion")

    out <= cond

  ; Ensure InputOnly is inlined by default.
  ; CHECK-NOT: module InputOnly(

  ; Ensure InputOnly is not inlined when disabled.
  ; DISABLEMODULE-LABEL: module InputOnly(

  module InputOnly:
    input clock : Clock
    input cond : UInt<1>
    assert(clock, cond, cond, "Some assertion")

  module Top:
    input clock : Clock
    input cond : UInt<1>
    output out : UInt<1>

    inst instance_extracted of InstanceExtracted
    instance_extracted.clock <= clock
    instance_extracted.cond <= cond
    out <= instance_extracted.out

    inst input_only of InputOnly
    input_only.clock <= clock
    input_only.cond <= cond

