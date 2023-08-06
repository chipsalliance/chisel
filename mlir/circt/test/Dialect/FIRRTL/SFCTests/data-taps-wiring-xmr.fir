; RUN: firtool %s --split-input-file | FileCheck %s

circuit SinkThroughWhens: %[[
  {
    "class":"sifive.enterprise.grandcentral.DataTapsAnnotation",
    "keys":[
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~SinkThroughWhens|SinkThroughWhens>val",
        "sink":"~SinkThroughWhens|SinkThroughWhens/mid:Middle/leaf:Leaf>tap"
      }
    ]
  },
  {
    "class":"firrtl.transforms.DontTouchAnnotation",
    "target":"~SinkThroughWhens|Leaf>tap"
  }
]]

; CHECK-LABEL: module Leaf
  module Leaf :
    input c: UInt<1>

    when c:
      wire tap : UInt<4>
      tap is invalid
      ; CHECK-NEXT: tap = 4'h3;
      ; CHECK-NEXT: endmodule

  module Middle :
    input c: UInt<1>
    
    when c:
      inst leaf of Leaf
      leaf.c <= c

  module SinkThroughWhens :
    input c: UInt<1>

    wire val : UInt<3>
    val <= UInt(3)

    inst mid of Middle
    mid.c <= c

; // -----

circuit TapWhenVal : %[[
  {
    "class":"sifive.enterprise.grandcentral.DataTapsAnnotation",
    "keys":[
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~TapWhenVal|TapWhenVal>val",
        "sink":"~TapWhenVal|TapWhenVal>tap"
      }
    ]
  }
]]
; CHECK-LABEL: module TapWhenVal
  module TapWhenVal :
    input c: UInt<1>
    output o: UInt<1>

    wire tap: UInt<3>
    tap is invalid

    when c:
      wire val : UInt<3>
      val <= UInt(1)

    o <= tap
; CHECK: assign o = 1'h1;

; // -----

circuit TapIntoWhen : %[[
  {
    "class":"sifive.enterprise.grandcentral.DataTapsAnnotation",
    "keys":[
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~TapIntoWhen|TapIntoWhen>val",
        "sink":"~TapIntoWhen|TapIntoWhen>tap"
      }
    ]
  }
]]
; CHECK-LABEL: module TapIntoWhen
  module TapIntoWhen :
    input c: UInt<1>
    output o: UInt<1>

    wire val : UInt<3>
    val <= UInt(3)

    o is invalid

    when c:
      wire tap: UInt
      tap is invalid
      o <= tap
; CHECK: assign  o = 1'h1;
