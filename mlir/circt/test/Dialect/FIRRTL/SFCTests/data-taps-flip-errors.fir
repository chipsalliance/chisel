; RUN: firtool --verilog %s --verify-diagnostics --split-input-file

; Test tapping a bundle with flips, sink also has flips.
; Tap sink must be passive.
circuit Top : %[[
  {
    "class": "sifive.enterprise.grandcentral.DataTapsAnnotation",
    "keys": [
      {
        "class": "sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source": "~Top|Top/c:Child>x",
        "sink": "~Top|Top>sink"
      }
    ]
  }
]]
  module Child :
    input x: {a : UInt<2>, flip b: UInt<2>}
    x.b <= UInt<2>(2)

  module Top :
    input x: {a : UInt<2>, flip b: UInt<2>}
    output tap : {a : UInt<2>, flip b: UInt<2>}

    inst c of Child
    c.x <= x

    ; expected-error @below {{Wiring Problem sink type "'!firrtl.bundle<a: uint<2>, b flip: uint<2>>'" must be passive (no flips) when using references}}
    wire sink : {a : UInt<2>, flip b: UInt<2>}
    tap <= sink



