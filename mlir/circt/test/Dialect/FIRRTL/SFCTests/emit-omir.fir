; RUN: firtool --repl-seq-mem --repl-seq-mem-file="dummy" --emit-omir %s | FileCheck %s

circuit Foo : %[[
  {
    "class": "freechips.rocketchip.objectmodel.OMIRFileAnnotation",
    "filename": "omir.json"
  },
  {
    "class": "freechips.rocketchip.objectmodel.OMIRAnnotation",
    "nodes": [
      {
        "info": "",
        "id": "OMID:0",
        "fields": [
          {"info": "", "name": "a", "value": "OMReference:0"},
          {"info": "", "name": "b", "value": "OMBigInt:42"},
          {"info": "", "name": "c", "value": "OMLong:ff"},
          {"info": "", "name": "d", "value": "OMString:hello"},
          {"info": "", "name": "f", "value": "OMBigDecimal:10.5"},
          {"info": "", "name": "g", "value": "OMDeleted"},
          {"info": "", "name": "h", "value": "OMConstant:UInt<2>(\"h1\")"},
          {"info": "", "name": "i", "value": 42},
          {"info": "", "name": "j", "value": true},
          {"info": "", "name": "k", "value": 3.14}
        ]
      },
      {
        "info": "",
        "id": "OMID:1",
        "fields": [
          {"info": "", "name": "a", "value": "OMReferenceTarget:~Foo|Foo"},
          {"info": "", "name": "b", "value": "OMInstanceTarget:~Foo|Foo"},
          {"info": "", "name": "c", "value": "OMMemberReferenceTarget:~Foo|Foo"},
          {"info": "", "name": "d", "value": "OMMemberInstanceTarget:~Foo|Foo"},
          {"info": "", "name": "e", "value": "OMDontTouchedReferenceTarget:~Foo|Foo"},
          {"info": "", "name": "f", "value": "OMReferenceTarget:~Foo|Bar"},
          {"info": "", "name": "g", "value": "OMReferenceTarget:~Foo|Foo>signed"}
        ]
      },
      {
        "info": "",
        "id": "OMID:2",
        "fields": [
          {"info": "", "name": "omType", "value": ["OMString:OMLazyModule", "OMString:OMSRAM"]},
          {"info": "", "name": "finalPath", "value": "OMMemberInstanceTarget:~Foo|Bar>longint"}
        ]
      },
      {
        "info": "",
        "id": "OMID:3",
        "fields": [
          {"info": "", "name": "omType", "value": ["OMString:OMLazyModule", "OMString:OMSRAM"]},
          {"info": "", "name": "finalPath", "value": "OMMemberReferenceTarget:~Foo|Bar>shortint"}
        ]
      },
      {
        "info": "",
        "id": "OMID:4",
        "fields": [
          {"info": "", "name": "containingModule", "value": "OMInstanceTarget:~Foo|Foo"},
          {"info": "", "name": "portField", "value": "OMReferenceTarget:~Foo|Foo/parameter:Bar>b.signed"}
        ]
      }
    ]
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Foo|Bar>shortint"
  }
]]
  extmodule MySRAM:
    defname = MySRAM
  module Foo :
    input unsigned : UInt<29>
    input zeroW : UInt<0>
    output signed : UInt<31>
    inst parameter of Bar
    parameter.b.signed <= unsigned
    parameter.b.qux <= unsigned
    signed <= unsigned
  module Bar :
    input b: {signed: UInt<1>, qux: UInt<1>}
    inst longint of MySRAM
    mem shortint :
        data-type => UInt<42>
        depth => 8
        read-latency => 1
        write-latency => 1
        writer => port
        read-under-write => undefined
    shortint.port is invalid

; CHECK-LABEL: FILE "omir.json"

; CHECK:       "id": "OMID:0"
; CHECK:       "name": "a"
; CHECK-NEXT:  "value": "OMReference:0"
; CHECK:       "name": "b"
; CHECK-NEXT:  "value": "OMBigInt:42"
; CHECK:       "name": "c"
; CHECK-NEXT:  "value": "OMLong:ff"
; CHECK:       "name": "d"
; CHECK-NEXT:  "value": "OMString:hello"
; CHECK:       "name": "f"
; CHECK-NEXT:  "value": "OMBigDecimal:10.5"
; CHECK:       "name": "g"
; CHECK-NEXT:  "value": "OMDeleted"
; CHECK:       "name": "h"
; CHECK-NEXT:  "value": "OMConstant:UInt<2>(\"h1\")"
; CHECK:       "name": "i"
; CHECK-NEXT:  "value": 42
; CHECK:       "name": "j"
; CHECK-NEXT:  "value": true
; CHECK:       "name": "k"
; CHECK-NEXT:  "value": 3.14

; CHECK:       "id": "OMID:1"
; CHECK:       "name": "a"
; CHECK-NEXT:  "value": "OMReferenceTarget:~Foo|Foo"
; CHECK:       "name": "b"
; CHECK-NEXT:  "value": "OMInstanceTarget:~Foo|Foo"
; CHECK:       "name": "c"
; CHECK-NEXT:  "value": "OMMemberReferenceTarget:~Foo|Foo"
; CHECK:       "name": "d"
; CHECK-NEXT:  "value": "OMMemberInstanceTarget:~Foo|Foo"
; CHECK:       "name": "e"
; CHECK-NEXT:  "value": "OMDontTouchedReferenceTarget:~Foo|Foo"
; CHECK:       "name": "f"
; CHECK-NEXT:  "value": "OMReferenceTarget:~Foo|Bar"
; CHECK:       "name": "g",
; CHECK-NEXT:  "value": "OMReferenceTarget:~Foo|Foo>signed_0"

; CHECK:       "id": "OMID:2"
; CHECK:       "name": "finalPath"
; CHECK-NEXT:  "value": "OMMemberInstanceTarget:~Foo|Foo/parameter_0:Bar/longint_0:MySRAM"

; CHECK:       "id": "OMID:3"
; CHECK:       "name": "finalPath"
; CHECK-NEXT:  "value": "OMMemberInstanceTarget:~Foo|Foo/parameter_0:Bar/shortint_0:shortint_0/shortint_ext:{{[^"]+}}"

; CHECK:       "id": "OMID:4"
; CHECK:       "name": "containingModule"
; CHECK-NEXT:  "value": "OMInstanceTarget:~Foo|Foo"
; CHECK:        "name": "portField",
; CHECK-NEXT:   "value": "OMReferenceTarget:~Foo|Foo/parameter_0:Bar>b_signed"
; CHECK:       "name": "ports"
; CHECK-NEXT:  "value": [
; CHECK-NEXT:    {
; CHECK-NEXT:      "ref": "OMDontTouchedReferenceTarget:~Foo|Foo>unsigned_0",
; CHECK-NEXT:      "direction": "OMString:Input",
; CHECK-NEXT:      "width": "OMBigInt:1d"
; CHECK-NEXT:    }
; CHECK-NEXT:    {
; CHECK-NEXT:      "ref": "OMDontTouchedReferenceTarget:~Foo|Foo>signed_0",
; CHECK-NEXT:      "direction": "OMString:Output",
; CHECK-NEXT:      "width": "OMBigInt:1f"
; CHECK-NEXT:    }
; CHECK-NEXT:  ]
