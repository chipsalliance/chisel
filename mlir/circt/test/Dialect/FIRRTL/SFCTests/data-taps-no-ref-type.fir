; RUN: firtool -lower-annotations-no-ref-type-ports %s | FileCheck %s

circuit Foo: %[[
  {
    "class": "sifive.enterprise.grandcentral.DataTapsAnnotation",
    "keys": [
      {
        "class": "sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source": "~Foo|Foo/bar:Bar>a",
        "sink": "~Foo|Foo/baz:Baz>a"
      }
    ]
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Foo|Bar>a"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Foo|Baz>a"
  }
]]
  module Bar:
    wire a: UInt<1>
    a is invalid

  module Baz:
    wire a: UInt<1>
    a is invalid

  module Foo:

    inst bar of Bar
    inst baz of Baz

    ; CHECK:      module Bar
    ; CHECK-NEXT:   output

    ; CHECK:      module Baz
    ; CHECK-NEXT:   input
