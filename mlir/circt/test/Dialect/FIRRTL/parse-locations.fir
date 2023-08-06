; RUN: circt-translate -import-firrtl --mlir-print-debuginfo -verify-diagnostics -mlir-print-local-scope %s | FileCheck %s

circuit MyModule :  @[CIRCUIT.scala 127]

  ; CHECK-LABEL: firrtl.module @MyModule(
  ; CHECK-SAME:    in %in: !firrtl.uint loc("InputPort.scala":0:0)
  ; CHECK-SAME:    out %out: !firrtl.uint<8> loc("OutputPort.scala":0:0)
  ; CHECK-SAME:  ) {
  module MyModule :   @[FooBar.scala 369:27]
    input in: UInt      @[InputPort.scala 0:0]
    output out: UInt<8> @[OutputPort.scala 0:0]

    ; CHECK: firrtl.connect {{.*}}loc("Somewhere.galaxy":42:1)
    out <= in   @[Somewhere.galaxy 42:1]

    ; CHECK: firrtl.node {{.*}}loc("Somewhere.galaxy":42:2)
    node ntest = in  @[Somewhere.galaxy 42:{2,2}]

    ; CHECK: firrtl.skip loc("SKipLoc.scala":42:24)
    skip  @[SKipLoc.scala 42:24]

  ; MODULE: CHECK: } loc("FooBar.scala":369:27)


  ; CHECK-LABEL: firrtl.module private @M2
  module M2 :   @[Filename With Space.scala 1:2]
    input in: UInt
  ; CHECK: } loc("Filename With Space.scala":1:2)

  ; CHECK-LABEL: firrtl.module private @M3
  module M3 :   @[NoSpaceInLocator]   ; expected-warning {{ignoring unknown @ info record format}}
    input in: UInt
  ; CHECK: } loc({{.*}}test{{.*}}Dialect{{.*}}FIRRTL{{.*}}parse-locations.fir"


  ; CHECK-LABEL: firrtl.module private @Subexpressions
  module Subexpressions :
    input in: UInt
    output auto : { out_0 : UInt<1> }
    ; CHECK: %0 = firrtl.subfield %auto[out_0] {{.*}} loc("Field":173:49)

    ; CHECK: %out_0 = firrtl.wire
    wire out_0 : { member : { 0 : { reset : UInt<1>}}}

    ; CHECK: %1 = firrtl.subfield %out_0[member] {{.*}} loc("Field":173:49)
    ; CHECK: %2 = firrtl.subfield %1["0"] {{.*}} loc("Field":173:49)
    ; CHECK: %3 = firrtl.subfield %2[reset] : {{.*}} loc("Field":173:49)
    ; CHECK: firrtl.strictconnect %0, %3 : {{.*}} loc("Field":173:49)
    auto.out_0 <- out_0.member.0.reset @[Field 173:49]

    ; Fused locators: https://github.com/llvm/circt/issues/224

    ; CHECK: %thing = firrtl.wire{{.*}} loc(fused["XX.scala":123:19, "YY.haskell":309:14, "ZZ.swift":3:4])
    wire thing : SInt<4> @[XX.scala 123:19 YY.haskell 309:14 ZZ.swift 3:4]

    ; CHECK: %other_thing = firrtl.wire{{.*}} loc("File with space.perl":1:23)
    wire other_thing : SInt<4> @[File with space.perl 1:23]

  ; CHECK-LABEL: firrtl.module private @issue1140
  ; https://github.com/llvm/circt/issues/1140
  module issue1140:
    input mask_bit_2 : UInt<1>
    node mask_nbit_2 = eq(mask_bit_2, UInt<1>("h0")) @[Misc.scala 210:20]

    ; Make sure the locator is applied to all the subexpressions.
    ; CHECK:  %c0_ui1 = firrtl.constant 0 : !firrtl.const.uint<1> loc("Misc.scala":210:20)
    ; CHECK:  %0 = firrtl.eq %mask_bit_2, %c0_ui1 : (!firrtl.uint<1>, !firrtl.const.uint<1>) -> !firrtl.uint<1> loc("Misc.scala":210:20)
    ; CHECK:  %mask_nbit_2 = firrtl.node interesting_name %0  : !firrtl.uint<1> loc("Misc.scala":210:20)



; CHECK: } loc("CIRCUIT.scala":127:0)
