; RUN: firtool --output-annotation-file=Baz.anno.json %s | FileCheck %s

; Test that a wire in an inlined instance get the proper Trace Annotation.
circuit Foo: %[[
  {
    "class":"chisel3.experimental.Trace$TraceNameAnnotation",
    "target":"~Foo|Foo/bar:Bar/baz:Baz>_x",
    "chiselTarget":"~Foo|Foo/bar:Bar/baz:Baz>_x_v1"
  },
  {
    "class":"chisel3.experimental.Trace$TraceAnnotation",
    "target":"~Foo|Foo/bar:Bar/baz:Baz>_x",
    "chiselTarget":"~Foo|Foo/bar:Bar/baz:Baz>_x_v2"
  },
  {
    "class": "firrtl.passes.InlineAnnotation",
    "target": "~Foo|Bar"
  }
]]
  module Baz:
    input a: UInt<1>
    output b: UInt<1>

    node _x = a
    b <= _x

  module Bar:
    input a: UInt<1>
    output b: UInt<1>

    inst baz of Baz
    baz.a <= a
    b <= baz.b

  module Foo:
    input a: UInt<1>
    output b: UInt<1>

    inst bar of Bar

    bar.a <= a
    b <= bar.b


; Wire "_x" should not be optimized away.
;
; CHECK-LABEL: module Baz
; CHECK:         wire [[_x:[A-Za-z0-9_]+]]

; CHECK-LABEL: module Foo
; CHECK:         Baz [[bar_baz:[a-zA-Z0-9_]+]]

; The final Annotation, in either the original TraceNameAnnotation or
; TraceAnnotation variant the wire name.  The former is serialized as a
; TraceAnnotation.  This is due to this version of the annotation being
; deprecated in upstream Chisel.  Chisel APIs have been updated to support
; reading both of these and treating them as the same.
;
; CHECK-LABEL: FILE "Baz.anno.json"
; CHECK:         "class": "chisel3.experimental.Trace$TraceAnnotation",
; CHECK-NEXT:    "target": "~Foo|Foo/[[bar_baz]]:Baz>[[_x]]",
; CHECK-NEXT:    "chiselTarget": "~Foo|Foo/bar:Bar/baz:Baz>_x_v1"
; CHECK:         "class": "chisel3.experimental.Trace$TraceAnnotation",
; CHECK-NEXT:    "target": "~Foo|Foo/[[bar_baz]]:Baz>[[_x]]",
; CHECK-NEXT:    "chiselTarget": "~Foo|Foo/bar:Bar/baz:Baz>_x_v2"
