; RUN: firtool --split-input-file %s | FileCheck %s

; A number of tests use an invalidated wire with a DontTouchAnnotation to block
; constant sinking into the XMR.  This is to ensure that the test can check a
; valid XMR as opposed to "assign foo = 1'h0".

; This test was extracted from:
; - github.com/sifive/$internal:
;   - src/test/scala/grandcentral/DataTapsTest.scala
circuit Top : %[[
  {
    "class":"sifive.enterprise.grandcentral.DataTapsAnnotation",
    "keys":[
      {
        "class":"sifive.enterprise.grandcentral.DataTapModuleSignalKey",
        "module":"~Top|BlackBox",
        "internalPath":"foo.bar.in",
        "sink":"~Top|Top>tap_0"
      },
      {
        "class":"sifive.enterprise.grandcentral.DataTapModuleSignalKey",
        "module":"~Top|BlackBox",
        "internalPath":"foo.bar.out",
        "sink":"~Top|Top>tap_1"
      }
    ]
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Top>in"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Top>out"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Top>tap_0"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Top>tap_1"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Top>tap_2"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Child>always"
  },
  {
    "class":"sifive.enterprise.grandcentral.MemTapAnnotation",
    "source": "~Top|Child>always",
    "sink":["~Top|Top>tap_2[0]", "~Top|Top>tap_2[1]"]
  }
]]
  extmodule BlackBox :
    input in : UInt<1>
    output out : UInt<1>
    defname = BlackBox

  module Child :
    input clock : Clock
    input reset : Reset
    output io : { flip in : UInt<1>, out : UInt<1>}

    inst localparam of BlackBox
    localparam.out is invalid
    localparam.in is invalid
    localparam.in <= io.in
    io.out <= localparam.out

    cmem always : UInt<1>[2]

    infer mport w = always[io.in], clock
    w <= io.in

    ; mem always :
    ;   data-type => UInt<1>
    ;   depth => 2
    ;   read-latency => 0
    ;   write-latency => 1
    ;   writer => w
    ;   read-under-write => undefined
    ; always.w.clk <= clock
    ; always.w.en <= UInt<1>(1)
    ; always.w.addr <= io.in
    ; always.w.data <= io.in
    ; always.w.mask <= UInt<1>(1)

  module ChildWrapper :
    input clock : Clock
    input reset : Reset
    output io : { flip in : UInt<1>, out : UInt<1>}

    inst signed of Child
    signed.clock <= clock
    signed.reset <= reset
    signed.io.in <= io.in
    io.out <= signed.io.out

  module Top :
    input clock : Clock
    input reset : UInt<1>
    output io : { flip in : UInt<1>, out : UInt<1>}

    inst unsigned of ChildWrapper
    unsigned.clock <= clock
    unsigned.reset <= reset
    wire in : UInt<1>
    wire out : UInt<1>
    node _child_io_in_T = and(io.in, in)
    unsigned.io.in <= _child_io_in_T
    node _io_out_T = and(unsigned.io.out, out)
    io.out <= _io_out_T

    wire tap_0: UInt<1>
    tap_0 is invalid

    wire tap_1: UInt<1>
    tap_1 is invalid

    wire tap_2: UInt<1>[2]
    tap_2 is invalid

    in <= or(tap_0, tap_2[0])
    out <= or(tap_1, tap_2[1])

; CHECK:      module Top
; CHECK:        tap_0 = Top.unsigned_0.signed_0.localparam_0.foo.bar.in;
; CHECK-NEXT:   tap_1 = Top.unsigned_0.signed_0.localparam_0.foo.bar.out;
; CHECK-NEXT:   tap_2_0 = Top.unsigned_0.signed_0.always_ext.Memory[0];
; CHECK-NEXT:   tap_2_1 = Top.unsigned_0.signed_0.always_ext.Memory[1];

; // -----

circuit Top : %[[
  {
    "class": "sifive.enterprise.grandcentral.DataTapsAnnotation",
    "keys": [
      {
        "class": "sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source": "~Top|Top/foo:Foo>f",
        "sink": "~Top|Top>b"
      },
      {
        "class": "sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source": "~Top|Top/foo:Foo>g",
        "sink": "~Top|Top>c"
      },
      {
        "class": "sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source": "~Top|Top>k",
        "sink": "~Top|Top>d"
      }
    ]
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Foo>inv"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Top>inv"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Foo>g"
  }
]]
  module Foo :
    output g : UInt<1>
    wire inv: UInt<1>
    inv is invalid
    wire f : UInt<1>
    g <= inv
    f <= inv

  module Top:
    output io : { b : UInt<1>, c : UInt<1>, d: UInt<1> }

    inst foo of Foo
    wire inv: UInt<1>
    inv is invalid
    wire k : UInt<1>
    k <= inv

    wire b: UInt<1>
    b is invalid

    wire c: UInt<1>
    c is invalid

    wire d: UInt<1>
    d is invalid

    io.b <= b
    io.c <= c
    io.d <= d

    ; CHECK:      module Top
    ; CHECK:        io_b = Top.foo.f_probe;
    ; CHECK-NEXT:   io_c = Top.foo.g_probe;
    ; CHECK-NEXT:   io_d = inv;

; // -----

circuit TestHarness : %[[
  {
    "class":"sifive.enterprise.grandcentral.DataTapsAnnotation",
    "keys":[
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~TestHarness|TestHarness/system:Top/test:Test>signal",
        "sink":"~TestHarness|Companion>tap_0"
      }
    ]
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~TestHarness|Test>inv"
  }
]]
  module TestHarness :
    output io : UInt<1>
    inst system of Top
    io <= system.io
  module Top :
    output io : UInt<1>
    inst test of Test
    inst Companion of Companion
    io <= Companion.io
  module Companion :
    output io : UInt<1>
    wire tap_0: UInt<8>
    tap_0 is invalid

    io <= tap_0
  module Test :
    wire inv: UInt<1>
    inv is invalid
    wire signal : UInt<8>
    signal <= inv

; CHECK:       module Top(
; CHECK-NOT:   endmodule
; CHECK:         Companion Companion (
; CHECK-NEXT:    .[[bore:[a-zA-Z0-9_]+]] (Top.test.signal_probe)
; CHECK:       endmodule

; CHECK:       module Companion(
; CHECK-NOT:   endmodule
; CHECK-NEXT:    input [7:0] [[bore]]
; CHECK:         io = [[bore]][0];
; CHECK:       endmodule

; // -----

circuit Top : %[[
  {
    "class":"sifive.enterprise.grandcentral.DataTapsAnnotation",
    "keys":[
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT/submodule:Submodule>wire_Submodule",
        "sink":"~Top|Submodule>tap_0"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT>wire_DUT",
        "sink":"~Top|Submodule>tap_1"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top>wire_Top",
        "sink":"~Top|Submodule>tap_2"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT/submodule:Submodule>port_Submodule",
        "sink":"~Top|Submodule>tap_3"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT>port_DUT",
        "sink":"~Top|Submodule>tap_4"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top>port_Top",
        "sink":"~Top|Submodule>tap_5"
      }
    ]
  },
  {
    "class":"sifive.enterprise.grandcentral.DataTapsAnnotation",
    "keys":[
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT/submodule:Submodule>wire_Submodule",
        "sink":"~Top|DUT>tap_6"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT>wire_DUT",
        "sink":"~Top|DUT>tap_7"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top>wire_Top",
        "sink":"~Top|DUT>tap_8"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT/submodule:Submodule>port_Submodule",
        "sink":"~Top|DUT>tap_9"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT>port_DUT",
        "sink":"~Top|DUT>tap_10"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top>port_Top",
        "sink":"~Top|DUT>tap_11"
      }
    ]
  },
  {
    "class":"sifive.enterprise.grandcentral.DataTapsAnnotation",
    "keys":[
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT/submodule:Submodule>wire_Submodule",
        "sink":"~Top|Top>tap_12[0]"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT>wire_DUT",
        "sink":"~Top|Top>tap_12[1]"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top>wire_Top",
        "sink":"~Top|Top>tap_12[2]"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT/submodule:Submodule>port_Submodule",
        "sink":"~Top|Top>tap_12[3]"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT>port_DUT",
        "sink":"~Top|Top>tap_12[4]"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top>port_Top",
        "sink":"~Top|Top>tap_12[5]"
      }
    ]
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Submodule>inv"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|DUT>inv"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Top>inv"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Submodule>tap_0"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Submodule>tap_1"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Submodule>tap_2"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Submodule>tap_3"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Submodule>tap_4"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Submodule>tap_5"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|DUT>tap_6"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|DUT>tap_7"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|DUT>tap_8"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|DUT>tap_9"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|DUT>tap_10"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|DUT>tap_11"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Top>tap_12"
  }
]]
  module Submodule :
    output port_Submodule: UInt<1>
    port_Submodule is invalid

    wire inv: UInt<1>
    inv is invalid

    wire wire_Submodule: UInt<1>
    wire_Submodule <= inv

    wire tap_0 : UInt<1>
    wire tap_1 : UInt<1>
    wire tap_2 : UInt<1>
    wire tap_3 : UInt<1>
    wire tap_4 : UInt<1>
    wire tap_5 : UInt<1>
    tap_0 is invalid
    tap_1 is invalid
    tap_2 is invalid
    tap_3 is invalid
    tap_4 is invalid
    tap_5 is invalid

  module DUT :
    output port_DUT: UInt<1>
    port_DUT is invalid

    wire inv: UInt<1>
    inv is invalid

    wire wire_DUT: UInt<1>
    wire_DUT <= inv

    inst submodule of Submodule

    wire tap_6 : UInt<1>
    wire tap_7 : UInt<1>
    wire tap_8 : UInt<1>
    wire tap_9 : UInt<1>
    wire tap_10 : UInt<1>
    wire tap_11 : UInt<1>
    tap_6 is invalid
    tap_7 is invalid
    tap_8 is invalid
    tap_9 is invalid
    tap_10 is invalid
    tap_11 is invalid

  module Top :
    output port_Top : UInt<1>
    port_Top is invalid

    wire inv: UInt<1>
    inv is invalid

    wire wire_Top: UInt<1>
    wire_Top <= inv

    inst dut of DUT
    wire tap_12 : UInt<1>[6]
    tap_12 is invalid

; CHECK-LABEL: module Submodule
; CHECK-NEXT:    input [[Submodule_tap_1_port:[a-zA-Z0-9_]+]]
; CHECK-NEXT:          [[Submodule_tap_2_port:[a-zA-Z0-9_]+]]
; CHECK-NEXT:          [[Submodule_tap_5_port:[a-zA-Z0-9_]+]]
;
; CHECK-DAG:     tap_0 = inv;
; CHECK-DAG:     tap_1 = [[Submodule_tap_1_port]];
; CHECK-DAG:     tap_2 = [[Submodule_tap_2_port]];
; CHECK-DAG:     tap_3 = 1'h0;
; CHECK-DAG:     tap_4 = 1'h0;
; CHECK-DAG:     tap_5 = [[Submodule_tap_5_port]];

; CHECK-LABEL: module DUT
; CHECK-NEXT:    input [[DUT_tap_2_port:[a-zA-Z0-9_]+]]
; CHECK-NEXT:          [[DUT_tap_5_port:[a-zA-Z0-9_]+]]
; CHECK-NEXT:          [[DUT_tap_8_port:[a-zA-Z0-9_]+]]
; CHECK-NEXT:          [[DUT_tap_11_port:[a-zA-Z0-9_]+]]
;
; CHECK-DAG:     tap_6 = DUT.submodule.wire_Submodule_probe;
; CHECK-DAG:     tap_7 = inv;
; CHECK-DAG:     tap_8 = [[DUT_tap_8_port]];
; CHECK-DAG:     tap_9 = 1'h0;
; CHECK-DAG:     tap_10 = 1'h0;
; CHECK-DAG:     tap_11 = [[DUT_tap_11_port]];
;
; CHECK:         Submodule submodule (
; CHECK-DAG:       .[[Submodule_tap_1_port]] (inv)
; CHECK-DAG:       .[[Submodule_tap_2_port]] ([[DUT_tap_2_port]])
; CHECK-DAG:       .[[Submodule_tap_5_port]] ([[DUT_tap_5_port]])

; CHECK-LABEL: module Top
;
; CHECK-DAG:     tap_12_0 = Top.dut.submodule.wire_Submodule
; CHECK-DAG:     tap_12_1 = Top.dut.wire_DUT
; CHECK-DAG:     tap_12_2 = inv;
; CHECK-DAG:     tap_12_3 = 1'h0
; CHECK-DAG:     tap_12_4 = 1'h0
; CHECK-DAG:     tap_12_5 = 1'h0
;
; CHECK:         DUT dut (
; CHECK-DAG:       .[[DUT_tap_2_port]] (inv)
; CHECK-DAG:       .[[DUT_tap_5_port]] (1'h0)
; CHECK-DAG:       .[[DUT_tap_8_port]] (inv)
; CHECK-DAG:       .[[DUT_tap_11_port]] (1'h0)

; // -----

circuit Top : %[[
  {
    "class":"firrtl.transforms.DontTouchAnnotation",
    "target":"~Top|Child>sum"
  },
  {
    "class":"sifive.enterprise.grandcentral.DataTapsAnnotation",
    "keys":[
      {
        "class":"sifive.enterprise.grandcentral.DataTapModuleSignalKey",
        "module":"~Top|Child",
        "internalPath":"sum",
        "sink":"~Top|Top>dt._0"
      },
      {
        "class":"sifive.enterprise.grandcentral.DataTapModuleSignalKey",
        "module":"~Top|Child",
        "internalPath":"not.a.real.path",
        "sink":"~Top|Top>dt._1"
      }
    ]
  }
]]
  module Child :
    output io : { flip in : UInt<8>, out : UInt<8>}

    node sum = tail(add(io.in, UInt<1>(1)), 1)

    io.out <= sum

  module Top :
    output io : { flip in : UInt<8>, out : UInt<8>}
    output taps : UInt<8>[2]

    inst child of Child
    io <= child.io

    wire dt: {_0: UInt<8>, _1: UInt<8>}
    dt is invalid

    taps[0] <= dt._0
    taps[1] <= dt._1

; CHECK:     module Top(
; CHECK-NOT: endmodule
; CHECK:       taps_0 = Top.child.sum;
; CHECK-NEXT:  taps_1 = Top.child.not.a.real.path;
; CHECK:     endmodule
