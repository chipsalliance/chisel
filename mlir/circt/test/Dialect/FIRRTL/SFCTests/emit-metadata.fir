; RUN: firtool --verify-diagnostics --verilog %s | FileCheck %s

circuit Foo : %[[{
  "class": "firrtl.transforms.BlackBoxInlineAnno",
  "name": "hello.v",
  "text": "// world",
  "target": "~Foo|Foo"
},{
  "class": "sifive.enterprise.firrtl.SitestBlackBoxAnnotation",
  "filename": "blackboxes.json"
}]]
  ; An inline blackbox should not appear in the generated metadata.
  extmodule Foo :
    input clock : Clock
    defname = Foo

; CHECK-LABEL: FILE "blackboxes.json"
; CHECK-EMPTY:
; CHECK-NEXT: []
