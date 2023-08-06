; RUN: firtool -split-input-file -ir-fir %s -verify-diagnostics
; Tests extracted from:
;   - test/scala/firrtlTests/transforms/DedupTests.scala

; "modules should not be deduped if the NoCircuitDedupAnnotation is supplied"
;
; CHECK-LABEL: firrtl.circuit "Top"
; expected-error @below {{Unhandled annotation: {class = "firrtl.transforms.NoCircuitDedupAnnotation"}}}
circuit Top: %[[
  {
    "class":"firrtl.transforms.NoCircuitDedupAnnotation"
  }
]]
  ; CHECK: firrtl.module @A(
  module A:
    input a: UInt<1>
    output b: UInt<1>
    b <= a
  ; CHECK: firrtl.module @A_(
  module A_:
    input a: UInt<1>
    output b: UInt<1>
    b <= a
  ; CHECK: firrtl.module @Top(
  module Top:
    input a: UInt<1>
    output b: UInt<1>

    ; CHECK-NEXT: firrtl.instance a1 @A(
    ; CHECK-NEXT: firrtl.instance a2 @A_(
    inst a1 of A
    inst a2 of A_
    a1.a <= a
    a2.a <= a
    b <= xor(a1.b, a2.b)
