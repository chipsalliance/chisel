; RUN: firtool --split-input-file %s --ir-fir | FileCheck %s
; Tests extracted from:
; - test/scala/firrtlTests/WidthSpec.scala

; Literal width checks
circuit Foo :
  ; CHECK-LABEL: firrtl.module @Foo(
  ; CHECK-SAME: out %si0: !firrtl.sint<3>
  ; CHECK-SAME: out %si1: !firrtl.sint<3>
  ; CHECK-SAME: out %si2: !firrtl.sint<2>
  ; CHECK-SAME: out %si3: !firrtl.sint<1>
  ; CHECK-SAME: out %si4: !firrtl.sint<1>
  ; CHECK-SAME: out %si5: !firrtl.sint<2>
  ; CHECK-SAME: out %si6: !firrtl.sint<3>
  ; CHECK-SAME: out %si7: !firrtl.sint<3>
  ; CHECK-SAME: out %si8: !firrtl.sint<4>
  ; CHECK-SAME: out %ui0: !firrtl.uint<1>
  ; CHECK-SAME: out %ui1: !firrtl.uint<1>
  ; CHECK-SAME: out %ui2: !firrtl.uint<2>
  ; CHECK-SAME: out %ui3: !firrtl.uint<2>
  ; CHECK-SAME: out %ui4: !firrtl.uint<3>
  module Foo :
    output si0 : SInt
    output si1 : SInt
    output si2 : SInt
    output si3 : SInt
    output si4 : SInt
    output si5 : SInt
    output si6 : SInt
    output si7 : SInt
    output si8 : SInt

    output ui0 : UInt
    output ui1 : UInt
    output ui2 : UInt
    output ui3 : UInt
    output ui4 : UInt

    si0 <= SInt(-4)
    si1 <= SInt(-3)
    si2 <= SInt(-2)
    si3 <= SInt(-1)
    si4 <= SInt(0)
    si5 <= SInt(1)
    si6 <= SInt(2)
    si7 <= SInt(3)
    si8 <= SInt(4)

    ui0 <= UInt(0)
    ui1 <= UInt(1)
    ui2 <= UInt(2)
    ui3 <= UInt(3)
    ui4 <= UInt(4)
