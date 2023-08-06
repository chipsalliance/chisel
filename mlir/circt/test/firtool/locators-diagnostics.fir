; RUN: firtool %s -parse-only -mlir-print-debuginfo --mlir-print-local-scope | FileCheck %s --check-prefixes=COMMON,PREFER
; RUN: firtool %s -parse-only -mlir-print-debuginfo --mlir-print-local-scope -fuse-info-locators | FileCheck %s --check-prefixes=COMMON,FUSE
; RUN: firtool %s -parse-only -mlir-print-debuginfo --mlir-print-local-scope -prefer-info-locators | FileCheck %s --check-prefixes=COMMON,PREFER
; RUN: firtool %s -parse-only -mlir-print-debuginfo --mlir-print-local-scope -ignore-info-locators | FileCheck %s --check-prefixes=COMMON,IGNORE

; RUN: not firtool %s -fuse-info-locators 2>&1 | FileCheck %s --check-prefixes=DIAG
; DIAG: path/to/Foo.scala:273:69: error:
; DIAG: locators-diagnostics.fir:20:5: note: additional location here
; DIAG: locators-diagnostics.fir:23:5: error:
; DIAG-NOT: note
circuit Error:
  module Error:
    output xa: UInt<1>
    output ya: UInt<1>

    ; COMMON: %x = firrtl.wire
    ; FUSE-SAME: loc(fused["path/to/Foo.scala":273:69, "{{.+}}.fir":20:5])
    ; PREFER-SAME: loc("path/to/Foo.scala":273:69)
    ; IGNORE-SAME: loc("{{.+}}.fir":20:5)
    wire x : { boop: { foo: { a: UInt<1>, b: UInt<2>}, bar: { a: UInt<1> }}} @[path/to/Foo.scala 273:69]
    ; COMMON: %y = firrtl.wire
    ; COMMON-SAME: loc("{{.+}}.fir":23:5)
    wire y : { boop: { foo: { a: UInt<1>, b: UInt<2>}, bar: { a: UInt<1> }}} ; (no locator)

    ; invalidate all but .boop.foo.a
    x.boop.bar is invalid
    x.boop.foo.b is invalid
    y.boop.bar is invalid
    y.boop.foo.b is invalid

    ; Connects
    xa <= x.boop.foo.a
    ya <= y.boop.foo.a
