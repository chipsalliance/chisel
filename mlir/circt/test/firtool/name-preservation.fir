; RUN: firtool --preserve-values=named %s | FileCheck %s 

circuit Foo:
  ; CHECK-LABEL: module Foo
  module Foo:
    input a: {a: UInt<1>, flip b: UInt<1>}
    output b: {a: UInt<1>, flip b: UInt<1>}

    ; Unnamed wires are always removed.
    ; CHECK-NOT: wire _x_a;
    ; CHECK-NOT: wire _x_b;

    wire _x: {a: UInt<1>, flip b: UInt<1>}
    _x <= a

    ; Default behavior is to preserve named wires.
    ; CHECK:        wire x_a
    ; CHECK:        wire x_b
    wire x: {a: UInt<1>, flip b: UInt<1>}
    x <= _x

    ; Unnamed nodes are always removed.
    ; CHECK-NOT: wire _y_b
    node _y_b = x.b

    ; Default behavior is to preserve named nodes.
    ; CHECK:        wire y
    node y = _y_b

    b.a <= y
    x.b <= b.b
