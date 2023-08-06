; RUN: firtool --split-input-file %s --ir-fir | FileCheck %s
; Tests extracted from:
; - test/scala/firrtlTests/AsyncResetSpec.scala

; Complex literals should be allowed as reset values for AsyncReset.
circuit Foo:
  module Foo:
    input clock : Clock
    input reset : AsyncReset
    input x : UInt<1>[4]
    output z : UInt<1>[4]
    wire literal : UInt<1>[4]
    literal[0] <= UInt<1>("h00")
    literal[1] <= UInt<1>("h00")
    literal[2] <= UInt<1>("h00")
    literal[3] <= UInt<1>("h00")
    ; CHECK: %r_0 = firrtl.regreset %clock, %reset, %c0_ui1
    ; CHECK: %r_1 = firrtl.regreset %clock, %reset, %c0_ui1
    ; CHECK: %r_2 = firrtl.regreset %clock, %reset, %c0_ui1
    ; CHECK: %r_3 = firrtl.regreset %clock, %reset, %c0_ui1
    reg r : UInt<1>[4], clock with : (reset => (reset, literal))
    r <= x
    z <= r

// -----

; Complex literals of complex literals should be allowed as reset values for
; AsyncReset.
circuit Foo:
  module Foo:
    input clock : Clock
    input reset : AsyncReset
    input x : UInt<1>[4]
    output z : UInt<1>[4]
    wire literal : UInt<1>[2]
    literal[0] <= UInt<1>("h01")
    literal[1] <= UInt<1>("h01")
    wire complex_literal : UInt<1>[4]
    complex_literal[0] <= literal[0]
    complex_literal[1] <= literal[1]
    complex_literal[2] <= UInt<1>("h00")
    complex_literal[3] <= UInt<1>("h00")
    ; CHECK: %r_0 = firrtl.regreset %clock, %reset, %c1_ui1
    ; CHECK: %r_1 = firrtl.regreset %clock, %reset, %c1_ui1
    ; CHECK: %r_2 = firrtl.regreset %clock, %reset, %c0_ui1
    ; CHECK: %r_3 = firrtl.regreset %clock, %reset, %c0_ui1
    reg r : UInt<1>[4], clock with : (reset => (reset, complex_literal))
    r <= x
    z <= r

// -----

; Literals of bundle literals should be allowed as reset values for AsyncReset.
circuit Foo:
  module Foo:
    input clock : Clock
    input reset : AsyncReset
    input x : UInt<1>[4]
    output z : UInt<1>[4]
    wire bundle : {a: UInt<1>, b: UInt<1>}
    bundle.a <= UInt<1>("h01")
    bundle.b <= UInt<1>("h01")
    wire complex_literal : UInt<1>[4]
    complex_literal[0] <= bundle.a
    complex_literal[1] <= bundle.b
    complex_literal[2] <= UInt<1>("h00")
    complex_literal[3] <= UInt<1>("h00")
    ; CHECK: %r_0 = firrtl.regreset %clock, %reset, %c1_ui1
    ; CHECK: %r_1 = firrtl.regreset %clock, %reset, %c1_ui1
    ; CHECK: %r_2 = firrtl.regreset %clock, %reset, %c0_ui1
    ; CHECK: %r_3 = firrtl.regreset %clock, %reset, %c0_ui1
    reg r : UInt<1>[4], clock with : (reset => (reset, complex_literal))
    r <= x
    z <= r

// -----

; Cast literals should be allowed as reset values for AsyncReset.
circuit Foo:
  module Foo:
    input clock : Clock
    input reset : AsyncReset
    input x : SInt<4>
    output y : SInt<4>
    output z : SInt<4>
    ; CHECK: %r = firrtl.regreset %clock, %reset, %c0_si1
    reg r : SInt<4>, clock with : (reset => (reset, asSInt(UInt(0))))
    r <= x
    wire w : SInt<4>
    ; CHECK: %r2 = firrtl.regreset %clock, %reset, %c-1_si4
    reg r2 : SInt<4>, clock with : (reset => (reset, w))
    r2 <= x
    node n = UInt("hf")
    w <= asSInt(n)
    y <= r2
    z <= r

// -----

; Unassigned asynchronously reset registers should properly constantprop.
circuit Foo:
  module Foo:
    input clock : Clock
    input reset : AsyncReset
    output z : UInt<1>[4]
    wire literal : UInt<1>[2]
    literal[0] <= UInt<1>("h01")
    literal[1] <= UInt<1>("h01")
    wire complex_literal : UInt<1>[4]
    complex_literal[0] <= literal[0]
    complex_literal[1] <= literal[1]
    complex_literal[2] <= UInt<1>("h00")
    complex_literal[3] <= UInt<1>("h00")
    reg r : UInt<1>[4], clock with : (reset => (reset, complex_literal))
    z <= r
    ; CHECK: firrtl.strictconnect %z_0, %c1_ui1
    ; CHECK: firrtl.strictconnect %z_1, %c1_ui1
    ; CHECK: firrtl.strictconnect %z_2, %c0_ui1
    ; CHECK: firrtl.strictconnect %z_3, %c0_ui1

// -----

; Constantly assigned asynchronously reset registers should properly
; constantprop.
circuit Foo:
  module Foo:
    input clock : Clock
    input reset : AsyncReset
    output z : UInt<1>
    reg r : UInt<1>, clock with : (reset => (reset, r))
    r <= UInt(0)
    z <= r
    ; CHECK: firrtl.strictconnect %z, %c0_ui1

// -----

; Constantly assigned and initialized asynchronously reset registers should
; properly constantprop.
circuit Foo:
  module Foo:
    input clock : Clock
    input reset : AsyncReset
    output z : UInt<1>
    reg r : UInt<1>, clock with : (reset => (reset, UInt(0)))
    r <= UInt(0)
    z <= r
    ; CHECK: firrtl.strictconnect %z, %c0_ui1
