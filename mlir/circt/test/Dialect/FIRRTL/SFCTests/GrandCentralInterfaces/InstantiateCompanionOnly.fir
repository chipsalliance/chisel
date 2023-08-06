; RUN: firtool %s --grand-central-instantiate-companion | FileCheck %s --check-prefix CHECK

circuit Foo : %[[
  {
    "class": "sifive.enterprise.grandcentral.GrandCentralView$SerializedViewAnnotation",
    "name": "bar_view",
    "companion": "~Foo|Companion",
    "parent": "~Foo|Foo",
    "view": {
      "class": "sifive.enterprise.grandcentral.AugmentedBundleType",
      "defName": "Bar_View",
      "elements": [
        {
          "name": "bar_in",
          "description": "bar in",
          "tpe": {
            "class": "sifive.enterprise.grandcentral.AugmentedGroundType",
            "ref": {
              "circuit": "Foo",
              "module": "Foo",
              "path": [],
              "ref": "a",
              "component": []
            },
            "tpe": {
              "class": "sifive.enterprise.grandcentral.GrandCentralView$UnknownGroundType$"
            }
          }
        }
      ]
    }
  },
  {
    "class": "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
    "directory": "Wire/firrtl/gct",
    "filename": "Wire/firrtl/bindings.sv"
  },
  {
    "class": "sifive.enterprise.grandcentral.GrandCentralHierarchyFileAnnotation",
    "filename": "gct.yaml"
  }
]]
  module Companion :
    input clock: Clock
    input a: UInt<1>

    assert(clock, a, UInt<1>(1), "hello")

  module Foo :
    input clock: Clock

    wire a: UInt<1>
    a is invalid

    inst companion of Companion
    companion.clock <= clock
    companion.a <= a

    ; CHECK:     module Foo
    ; CHECK-NOT: endmodule
    ; CHECK-NOT:   emitted as a bind statement
    ; CHECK:       Companion companion
    ; CHECK:     endmodule

; Verify that an empty yaml file is created.
; CHECK: FILE "gct.yaml"
; CHECK: []  
