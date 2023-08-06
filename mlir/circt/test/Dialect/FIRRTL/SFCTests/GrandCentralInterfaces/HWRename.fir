; RUN: firtool --annotation-file %S/HWRename.anno.json %s | FileCheck %s

circuit Top:
  module Companion :
    output io : { }

    wire _WIRE : UInt<1>
    _WIRE <= UInt<1>("h0")

  module DUT:
    input a: UInt<1>
    output b: UInt<1>

    wire signed: UInt<1>

    signed <= a
    b <= signed

    inst companion of Companion

  module Top:
    input a: UInt<1>
    output b: UInt<1>

    inst signed of DUT
    signed.a <= a
    b <= signed.b

    ; CHECK:      module Companion(
    ; CHECK:        input [[port:[a-zA-Z0-9_]+]]
    ; CHECK:      );
    ; CHECK:        MyInterface MyView();
    ; CHECK-NEXT:   assign MyView.[[elementName:.+]] = [[port]];
    ; CHECK-NEXT: endmodule

    ; CHECK:      module DUT
    ; CHECK:        input a
    ; CHECK:        output b
    ; CHECK:          Companion companion (
    ; CHECK-NEXT:       .[[port]] (a)
    ; CHECK-NEXT:     );
    ; Wire got optimized away!!
    ; CHECK:          assign b = a;
    ; CHECK:      endmodule

    ; CHECK:      module Top
    ; CHECK:        DUT [[dutName:.+]] (
    ; CHECK:      endmodule

    ; CHECK: interface MyInterface
    ; CHECK-NEXT:   logic [[elementName]]
    ; CHECK-NEXT: endinterface
