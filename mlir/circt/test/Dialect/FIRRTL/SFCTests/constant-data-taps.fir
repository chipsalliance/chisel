; RUN: firtool --verilog %s | FileCheck %s

circuit ConstantSinking : %[[
  {
    "class": "sifive.enterprise.grandcentral.DataTapsAnnotation",
    "keys": [
      {
        "class": "sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source": "~ConstantSinking|ConstantSinking>w",
        "sink": "~ConstantSinking|ConstantSinking>t"
      }
    ]
  },
  {
    "class":"firrtl.transforms.DontTouchAnnotation",
    "target":"~ConstantSinking|ConstantSinking>t"
  }
]]
  module ConstantSinking:
    output out : UInt<1>
    wire t : UInt<1>
    out <= t
    t is invalid
    node w = UInt<1>(1)

; CHECK: wire t = 1'h1;
