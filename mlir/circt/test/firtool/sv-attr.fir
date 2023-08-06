; RUN: firtool %s | FileCheck %s

circuit Foo: %[[{"class": "firrtl.AttributeAnnotation",
                 "description": "keep_hierarchy = \"true\"",
                 "target": "~Foo|Foo"},
                {"class": "firrtl.AttributeAnnotation",
                 "description": "mark_debug = \"yes\"",
                 "target": "~Foo|Foo>w"},
                {"class": "firrtl.transforms.DontTouchAnnotation",
                 "target": "~Foo|Foo>w"},
                {"class": "firrtl.AttributeAnnotation",
                 "description": "mark_debug = \"yes\"",
                 "target": "~Foo|Foo>n"},
                {"class": "firrtl.transforms.DontTouchAnnotation",
                 "target": "~Foo|Foo>n"},
                {"class": "firrtl.AttributeAnnotation",
                 "description": "keep = \"true\"",
                 "target": "~Foo|Foo>r"}]]
  ; CHECK: (* keep_hierarchy = "true" *)
  ; CHECK-NEXT: module Foo
  module Foo:
    input a: UInt<1>
    input clock: Clock
    output b1: UInt<1>
    output b2: UInt<1>
    ; CHECK: (* mark_debug = "yes" *)
    ; CHECK-NEXT: wire w
    wire w: UInt<1>
    ; CHECK:      (* mark_debug = "yes" *)
    ; CHECK-NEXT: wire n
    node n = w;
    ; CHECK:      (* keep = "true" *) 
    ; CHECK-NEXT: reg r
    reg r: UInt<1>, clock
    w <= a
    b1 <= n
    r <= a
    b2 <= r
