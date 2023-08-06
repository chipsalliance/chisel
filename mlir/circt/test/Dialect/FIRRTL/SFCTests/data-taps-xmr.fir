; RUN: firtool --verilog %s --split-input-file | FileCheck %s

circuit Top : %[[
  {
    "class": "sifive.enterprise.grandcentral.DataTapsAnnotation",
    "keys": [
      {
        "class": "sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source": "~Top|Top/foo:Foo/b:Bar>inv",
        "sink": "~Top|Top>tap"
      },
      {
        "class":"sifive.enterprise.grandcentral.DataTapModuleSignalKey",
        "module":"~Top|BlackBox",
        "internalPath":"random.something",
        "sink": "~Top|Top>tap2"
      },
      {
        "class":"sifive.enterprise.grandcentral.DataTapModuleSignalKey",
        "module":"~Top|BlackBox",
        "internalPath":"random.something_else",
        "sink": "~Top|Top>tap3"
      },
      {
        "class":"sifive.enterprise.grandcentral.DataTapModuleSignalKey",
        "module":"~Top|InternalPathChild",
        "internalPath":"io_out",
        "sink": "~Top|Top/unsigned:ChildWrapper/signed:Child>tap"
      }
    ]
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Top>tap"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Top>tap2"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Top>tap3"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Child>tap"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Bar>inv"
  }
]]
  module InternalPathChild :

    skip

  extmodule BlackBox :

    defname = BlackBox

  module Child :

    inst localparam of BlackBox
    localparam is invalid

    wire tap : UInt<1>
    tap is invalid

  module ChildWrapper :

    inst signed of Child

  module Bar :

    wire inv : UInt<2>
    inv is invalid

  module Foo :

    inst b of Bar

  module Top:

    inst foo of Foo

    wire tap : UInt<2>
    tap is invalid

    wire tap2 : UInt<4>
    tap2 is invalid

    wire tap3 : UInt<5>
    tap3 is invalid

    inst unsigned of ChildWrapper

    inst int of InternalPathChild

; CHECK-LABEL: module Child(
; CHECK-NOT:   endmodule
; CHECK:         input [[Child_boredPort:[a-zA-Z0-9_]+]]
; CHECK:         wire tap = [[Child_boredPort]];
; CHECK:       endmodule

; CHECK-LABEL: module ChildWrapper(
; CHECK-NOT:   endmodule
; CHECK:         input [[ChildWrapper_boredPort:[a-zA-Z0-9_]+]]
; CHECK:         Child signed_0 (
; CHECK-NEXT:      .[[Child_boredPort]] ([[ChildWrapper_boredPort]])
; CHECK:       endmodule

; CHECK-LABEL: module Top(
; CHECK-NOT:   endmodule
; CHECK:         tap = Top.foo.b.inv_probe;
; CHECK-NEXT:    tap2 = Top.unsigned_0.signed_0.localparam_0.random.something;
; CHECK-NEXT:    tap3 = Top.unsigned_0.signed_0.localparam_0.random.something_else;
; CHECK:         ChildWrapper unsigned_0 (
; CHECK-NEXT:      .[[ChildWrapper_boredPort]] (Top.int_0.io_out)
; CHECK:       endmodule

; // -----

; XMR references should work through 'when's.
; https://github.com/llvm/circt/issues/4334
circuit Top : %[[
  {
    "class":"sifive.enterprise.grandcentral.DataTapsAnnotation",
    "keys":[
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top>val",
        "sink":"~Top|Top/mid:Middle/leaf:Leaf>tap"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/mid:Middle/leaf:Leaf>val",
        "sink":"~Top|Top>tap"
      }
    ]
  }
]]
  module Leaf :
    input c: UInt<1>

    when c:
      wire val : UInt<3>
      val <= UInt(2)

      ; Tap Top's val
      wire tap : UInt<4>
      tap is invalid

  module Middle :
    input c: UInt<1>
    
    when c:
      inst leaf of Leaf
      leaf.c <= c

  ; CHECK-LABEL: module Top(
  ; CHECK-NEXT:    input c
  ; CHECK-NEXT:    output [2:0] o
  ; CHECK-NEXT:  );
  ; CHECK-EMPTY:
  ; CHECK-NEXT:    assign o = 3'h2;
  ; CHECK-NEXT:  endmodule
  module Top :
    input c: UInt<1>
    output o: UInt

    wire val : UInt<3>
    val <= UInt(1)

    inst mid of Middle
    mid.c <= c

    o is invalid

    ; Tap Leaf's val
    when c:
      wire tap : UInt
      tap is invalid
      o <= tap
