; RUN: firtool %s --split-input-file -verify-diagnostics

circuit TapBetweenWhens : %[[
  {
    "class":"sifive.enterprise.grandcentral.DataTapsAnnotation",
    "keys":[
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~TapBetweenWhens|TapBetweenWhens>val",
        "sink":"~TapBetweenWhens|TapBetweenWhens>tap"
      }
    ]
  }
]]
  module TapBetweenWhens :
    input c1: UInt<1>
    input c2: UInt<1>
    output o: UInt

    when c1:
      ; expected-error @below {{This value is involved with a Wiring Problem where the destination}}
      wire val : UInt<3>
      val <= UInt(1)

    o is invalid

    when c2:
      ; expected-note @below {{The destination is here.}}
      wire tap: UInt<3>
      tap is invalid
      o <= tap

; // -----

circuit NoSafeInsertionPoint : %[[
  {
    "class":"sifive.enterprise.grandcentral.DataTapsAnnotation",
    "keys":[
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~NoSafeInsertionPoint|NoSafeInsertionPoint>val",
        "sink":"~NoSafeInsertionPoint|NoSafeInsertionPoint>tap"
      }
    ]
  }
]]
  module NoSafeInsertionPoint :
    input c: UInt<1>
    output o: UInt<1>

    when c:
      ; expected-error @below {{This value is involved with a Wiring Problem where the destination}}
      wire val : UInt<3>
      val <= UInt(1)

    ; expected-note @below {{The destination is here.}}
    wire tap: UInt<3>
    tap is invalid
    o <= tap

; // -----

; expected-error @below {{Unable to apply annotation}}
circuit Top : %[[
  {
    "class": "firrtl.passes.wiring.SinkAnnotation",
    "target": "Top.BlackBox.sink",
    "pin": "test_pin"
  },
  {
    "class": "firrtl.passes.wiring.SourceAnnotation",
    "target": "Top.Top.legacy_tap",
    "pin": "test_pin"
  },
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

; expected-error @below {{'firrtl.extmodule' op cannot be used for both legacy Wiring and DataTaps simultaneously}}
  extmodule BlackBox :
    input sink : UInt<1>

    defname = BlackBox

  module Child :

    inst localparam of BlackBox
    localparam is invalid

    wire tap : UInt<1>
    tap is invalid

  module ChildWrapper :

    inst signed of Child

  module Bar :

    wire inv : UInt<1>
    inv is invalid

  module Foo :

    inst b of Bar

  module Top:

    inst foo of Foo

    wire tap : UInt<1>
    tap is invalid

    wire tap2 : UInt<1>
    tap2 is invalid

    wire tap3 : UInt<1>
    tap3 is invalid

    wire legacy_tap : UInt<1>
    legacy_tap is invalid

    inst unsigned of ChildWrapper

    inst int of InternalPathChild
