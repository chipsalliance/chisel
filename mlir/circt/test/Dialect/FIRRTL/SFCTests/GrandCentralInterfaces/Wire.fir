; RUN: firtool --lowering-options=emitBindComments --annotation-file %s.anno.json --annotation-file %s.Extract.anno.json %s | FileCheck %s --check-prefixes CHECK,EXTRACT
; RUN: firtool --lowering-options=emitBindComments --preserve-values=named --annotation-file %s.anno.json %s | FileCheck %s --check-prefixes CHECK,NOEXTRACT
; RUN: firtool --lowering-options=emitBindComments --annotation-file %s.anno.json --annotation-file %s.Extract.anno.json %s --annotation-file %s.YAML.anno.json | FileCheck %s --check-prefixes YAML
; RUN: firtool --lowering-options=emitBindComments --split-verilog --annotation-file %s.anno.json %s -o %t.folder > %t  && cat %t.folder/MyView_companion.sv  | FileCheck %s --check-prefixes MYVIEW_COMPANION
; RUN: firtool --lowering-options=emitBindComments --annotation-file %s.anno.json --annotation-file %s.Prefix.anno.json %s | FileCheck %s --check-prefixes PREFIX
; RUN: firtool --annotation-file %s.anno.json %s --disable-opt

circuit Top :
  extmodule BlackBox_DUT :
    input a : UInt<1>

  extmodule BlackBox_GCT :
    input a : UInt<1>

  extmodule BlackBox_DUTAndGCT :
    input a : UInt<1>

  module Submodule :
    input clock : Clock
    input reset : Reset
    input in : { uint : UInt<1>, vec : UInt<1>[2], multivec : UInt<1>[3][2], vecOfBundle : { uint : UInt<4>, sint : SInt<2>}[2], otherOther : { other : { uint : UInt<4>, sint : SInt<2>}}}
    output out : { uint : UInt<1>, vec : UInt<1>[2], multivec : UInt<1>[3][2], vecOfBundle : { uint : UInt<4>, sint : SInt<2>}[2], otherOther : { other : { uint : UInt<4>, sint : SInt<2>}}}

    wire w : { uint : UInt<1>, vec : UInt<1>[2], multivec : UInt<1>[3][2], vecOfBundle : { uint : UInt<4>, sint : SInt<2>}[2], otherOther : { other : { uint : UInt<4>, sint : SInt<2>}}}
    w.otherOther.other.sint <= in.otherOther.other.sint
    w.otherOther.other.uint <= in.otherOther.other.uint
    w.vecOfBundle[0].sint <= in.vecOfBundle[0].sint
    w.vecOfBundle[0].uint <= in.vecOfBundle[0].uint
    w.vecOfBundle[1].sint <= in.vecOfBundle[1].sint
    w.vecOfBundle[1].uint <= in.vecOfBundle[1].uint
    w.vec[0] <= in.vec[0]
    w.vec[1] <= in.vec[1]
    w.multivec[0][0] <= in.multivec[0][0]
    w.multivec[0][1] <= in.multivec[0][1]
    w.multivec[0][2] <= in.multivec[0][2]
    w.multivec[1][0] <= in.multivec[1][0]
    w.multivec[1][1] <= in.multivec[1][1]
    w.multivec[1][2] <= in.multivec[1][2]
    w.uint <= UInt<1>("h1")
    out.otherOther.other.sint <= w.otherOther.other.sint
    out.otherOther.other.uint <= w.otherOther.other.uint
    out.vecOfBundle[0].sint <= w.vecOfBundle[0].sint
    out.vecOfBundle[0].uint <= w.vecOfBundle[0].uint
    out.vecOfBundle[1].sint <= w.vecOfBundle[1].sint
    out.vecOfBundle[1].uint <= w.vecOfBundle[1].uint
    out.vec[0] <= w.vec[0]
    out.vec[1] <= w.vec[1]
    out.multivec[0][0] <= w.multivec[0][0]
    out.multivec[0][1] <= w.multivec[0][1]
    out.multivec[0][2] <= w.multivec[0][2]
    out.multivec[1][0] <= w.multivec[1][0]
    out.multivec[1][1] <= w.multivec[1][1]
    out.multivec[1][2] <= w.multivec[1][2]
    out.uint <= w.uint

  extmodule Tap:
    output clock: Clock
    output a: UInt<1>
    input b: UInt<1>

  extmodule ExtModuleWithPort:
    output source: UInt<1>[2]

  ; This checks that macro definitions (e.g. "define RANDOM") are included
  ; in MyView_companion.sv. There was a bug that macro definitions were not
  ; included in output files. Here, only "RANDOM" macro is checked.
  ; MYVIEW_COMPANION: define RANDOM
  module MyView_companion :
    output io : { }

    wire clock: Clock
    reg r: UInt<1>, clock

    inst bbox1 of BlackBox_GCT
    bbox1.a <= r
    inst bbox2 of BlackBox_DUTAndGCT
    bbox2.a <= r

    inst tap of Tap
    clock <= tap.clock
    r <= tap.a
    tap.b <= r

    wire _WIRE : UInt<1>
    _WIRE <= UInt<1>("h0")

  module DUT :
    input clock : Clock
    input reset : Reset
    input in : { uint : UInt<1>, vec : UInt<1>[2], multivec : UInt<1>[3][2], vecOfBundle : { uint : UInt<4>, sint : SInt<2>}[2], otherOther : { other : { uint : UInt<4>, sint : SInt<2>}}}
    output out : { uint : UInt<1>, vec : UInt<1>[2], multivec : UInt<1>[3][2], vecOfBundle : { uint : UInt<4>, sint : SInt<2>}[2], otherOther : { other : { uint : UInt<4>, sint : SInt<2>}}}

    wire w : { uint : UInt<1>, vec : UInt<1>[2], multivec : UInt<1>[3][2], vecOfBundle : { uint : UInt<4>, sint : SInt<2>}[2], otherOther : { other : { uint : UInt<4>, sint : SInt<2>}}}
    inst submodule of Submodule
    submodule.clock <= clock
    submodule.reset <= reset
    w.otherOther.other.sint <= in.otherOther.other.sint
    w.otherOther.other.uint <= in.otherOther.other.uint
    w.vecOfBundle[0].sint <= in.vecOfBundle[0].sint
    w.vecOfBundle[0].uint <= in.vecOfBundle[0].uint
    w.vecOfBundle[1].sint <= in.vecOfBundle[1].sint
    w.vecOfBundle[1].uint <= in.vecOfBundle[1].uint
    w.vec[0] <= in.vec[0]
    w.vec[1] <= in.vec[1]
    w.multivec[0][0] <= in.multivec[0][0]
    w.multivec[0][1] <= in.multivec[0][1]
    w.multivec[0][2] <= in.multivec[0][2]
    w.multivec[1][0] <= in.multivec[1][0]
    w.multivec[1][1] <= in.multivec[1][1]
    w.multivec[1][2] <= in.multivec[1][2]
    w.uint <= in.uint
    submodule.in.otherOther.other.sint <= w.otherOther.other.sint
    submodule.in.otherOther.other.uint <= w.otherOther.other.uint
    submodule.in.vecOfBundle[0].sint <= w.vecOfBundle[0].sint
    submodule.in.vecOfBundle[0].uint <= w.vecOfBundle[0].uint
    submodule.in.vecOfBundle[1].sint <= w.vecOfBundle[1].sint
    submodule.in.vecOfBundle[1].uint <= w.vecOfBundle[1].uint
    submodule.in.vec[0] <= w.vec[0]
    submodule.in.vec[1] <= w.vec[1]
    submodule.in.multivec[0][0] <= w.multivec[0][0]
    submodule.in.multivec[0][1] <= w.multivec[0][1]
    submodule.in.multivec[0][2] <= w.multivec[0][2]
    submodule.in.multivec[1][0] <= w.multivec[1][0]
    submodule.in.multivec[1][1] <= w.multivec[1][1]
    submodule.in.multivec[1][2] <= w.multivec[1][2]
    submodule.in.uint <= w.uint
    out.otherOther.other.sint <= submodule.out.otherOther.other.sint
    out.otherOther.other.uint <= submodule.out.otherOther.other.uint
    out.vecOfBundle[0].sint <= submodule.out.vecOfBundle[0].sint
    out.vecOfBundle[0].uint <= submodule.out.vecOfBundle[0].uint
    out.vecOfBundle[1].sint <= submodule.out.vecOfBundle[1].sint
    out.vecOfBundle[1].uint <= submodule.out.vecOfBundle[1].uint
    out.vec[0] <= submodule.out.vec[0]
    out.vec[1] <= submodule.out.vec[1]
    out.multivec[0][0] <= submodule.out.multivec[0][0]
    out.multivec[0][1] <= submodule.out.multivec[0][1]
    out.multivec[0][2] <= submodule.out.multivec[0][2]
    out.multivec[1][0] <= submodule.out.multivec[1][0]
    out.multivec[1][1] <= submodule.out.multivec[1][1]
    out.multivec[1][2] <= submodule.out.multivec[1][2]
    out.uint <= submodule.out.uint
    inst MyView_companion of MyView_companion

    inst bbox1 of BlackBox_DUT
    bbox1.a <= in.uint
    inst bbox2 of BlackBox_DUTAndGCT
    bbox2.a <= in.uint

    inst ext of ExtModuleWithPort

  module Top :
    input clock : Clock
    input reset : UInt<1>
    input in : { uint : UInt<1>, vec : UInt<1>[2], multivec : UInt<1>[3][2], vecOfBundle : { uint : UInt<4>, sint : SInt<2>}[2], otherOther : { other : { uint : UInt<4>, sint : SInt<2>}}}
    output out : { uint : UInt<1>, vec : UInt<1>[2], multivec : UInt<1>[3][2], vecOfBundle : { uint : UInt<4>, sint : SInt<2>}[2], otherOther : { other : { uint : UInt<4>, sint : SInt<2>}}}

    inst dut of DUT
    dut.clock <= clock
    dut.reset <= reset
    dut.in.otherOther.other.sint <= in.otherOther.other.sint
    dut.in.otherOther.other.uint <= in.otherOther.other.uint
    dut.in.vecOfBundle[0].sint <= in.vecOfBundle[0].sint
    dut.in.vecOfBundle[0].uint <= in.vecOfBundle[0].uint
    dut.in.vecOfBundle[1].sint <= in.vecOfBundle[1].sint
    dut.in.vecOfBundle[1].uint <= in.vecOfBundle[1].uint
    dut.in.vec[0] <= in.vec[0]
    dut.in.vec[1] <= in.vec[1]
    dut.in.multivec[0][0] <= in.multivec[0][0]
    dut.in.multivec[0][1] <= in.multivec[0][1]
    dut.in.multivec[0][2] <= in.multivec[0][2]
    dut.in.multivec[1][0] <= in.multivec[1][0]
    dut.in.multivec[1][1] <= in.multivec[1][1]
    dut.in.multivec[1][2] <= in.multivec[1][2]
    dut.in.uint <= in.uint
    out.otherOther.other.sint <= dut.out.otherOther.other.sint
    out.otherOther.other.uint <= dut.out.otherOther.other.uint
    out.vecOfBundle[0].sint <= dut.out.vecOfBundle[0].sint
    out.vecOfBundle[0].uint <= dut.out.vecOfBundle[0].uint
    out.vecOfBundle[1].sint <= dut.out.vecOfBundle[1].sint
    out.vecOfBundle[1].uint <= dut.out.vecOfBundle[1].uint
    out.vec[0] <= dut.out.vec[0]
    out.vec[1] <= dut.out.vec[1]
    out.multivec[0][0] <= dut.out.multivec[0][0]
    out.multivec[0][1] <= dut.out.multivec[0][1]
    out.multivec[0][2] <= dut.out.multivec[0][2]
    out.multivec[1][0] <= dut.out.multivec[1][0]
    out.multivec[1][1] <= dut.out.multivec[1][1]
    out.multivec[1][2] <= dut.out.multivec[1][2]
    out.uint <= dut.out.uint

    ; PREFIX:         module FOO_BAR_MyView_companion(
    ; PREFIX:         FOO_BAR_BAZ_MyInterface MyView(

    ; NOEXTRACT:      module MyView_companion(
    ; NOEXTRACT:      );
    ; NOEXTRACT:        MyInterface MyView();
    ; NOEXTRACT:        assign MyView.uint = [[DUT_w_uint:[a-zA-Z0-9_]+]];
    ; NOEXTRACT-NEXT:   assign MyView.vec[0] = [[DUT_w_vec_0:[a-zA-Z0-9_]+]];
    ; NOEXTRACT-NEXT:   assign MyView.vec[1] = [[DUT_w_vec_1:[a-zA-Z0-9_]+]];
    ; NOEXTRACT-NEXT:   assign MyView.multivec[0][0] = [[DUT_w_multivec_0_0:[a-zA-Z0-9_]+]];
    ; NOEXTRACT-NEXT:   assign MyView.multivec[0][1] = [[DUT_w_multivec_0_1:[a-zA-Z0-9_]+]];
    ; NOEXTRACT-NEXT:   assign MyView.multivec[0][2] = [[DUT_w_multivec_0_2:[a-zA-Z0-9_]+]];
    ; NOEXTRACT-NEXT:   assign MyView.multivec[1][0] = [[DUT_w_multivec_1_0:[a-zA-Z0-9_]+]];
    ; NOEXTRACT-NEXT:   assign MyView.multivec[1][1] = [[DUT_w_multivec_1_1:[a-zA-Z0-9_]+]];
    ; NOEXTRACT-NEXT:   assign MyView.multivec[1][2] = [[DUT_w_multivec_1_2:[a-zA-Z0-9_]+]];
    ; NOEXTRACT-NEXT:   assign MyView.vecOfBundle[0].sint = [[DUT_w_vecOfBundle_0_sint:[a-zA-Z0-9_]+]];
    ; NOEXTRACT-NEXT:   assign MyView.vecOfBundle[0].uint = [[DUT_w_vecOfBundle_0_uint:[a-zA-Z0-9_]+]];
    ; NOEXTRACT-NEXT:   assign MyView.vecOfBundle[1].sint = [[DUT_w_vecOfBundle_1_sint:[a-zA-Z0-9_]+]];
    ; NOEXTRACT-NEXT:   assign MyView.vecOfBundle[1].uint = [[DUT_w_vecOfBundle_1_uint:[a-zA-Z0-9_]+]];
    ; NOEXTRACT-NEXT:   assign MyView.otherOther.other.sint = [[DUT_w_otherOther_other_sint:[a-zA-Z0-9_]+]];
    ; NOEXTRACT-NEXT:   assign MyView.otherOther.other.uint = [[DUT_w_otherOther_other_uint:[a-zA-Z0-9_]+]];
    ; NOEXTRACT-NEXT:   assign MyView.sub_uint = 1'h1;
    ; NOEXTRACT-NEXT:   assign MyView.sub_vec[0] = [[DUT_submodule_w_vec_0:[a-zA-Z0-9_]+]];
    ; NOEXTRACT-NEXT:   assign MyView.sub_vec[1] = [[DUT_submodule_w_vec_1:[a-zA-Z0-9_]+]];
    ; NOEXTRACT-NEXT:   assign MyView.sub_vecOfBundle[0].sint = [[DUT_submodule_w_vecOfBundle_0_sint:[a-zA-Z0-9_]+]];
    ; NOEXTRACT-NEXT:   assign MyView.sub_vecOfBundle[0].uint = [[DUT_submodule_w_vecOfBundle_0_uint:[a-zA-Z0-9_]+]];
    ; NOEXTRACT-NEXT:   assign MyView.sub_vecOfBundle[1].sint = [[DUT_submodule_w_vecOfBundle_1_sint:[a-zA-Z0-9_]+]];
    ; NOEXTRACT-NEXT:   assign MyView.sub_vecOfBundle[1].uint = [[DUT_submodule_w_vecOfBundle_1_uint:[a-zA-Z0-9_]+]];
    ; NOEXTRACT-NEXT:   assign MyView.ext_port_1 = [[ExtModuleWithPort_source_1:[a-zA-Z0-9_]+]];
    ; NOEXTRACT:        Tap tap (
    ; NOEXTRACT-NEXT:     .b     (r),
    ; NOEXTRACT-NEXT:     .clock (clock),
    ; NOEXTRACT-NEXT:     .a     (_tap_a)
    ; NOEXTRACT-NEXT:   );
    ; NOEXTRACT-NEXT: endmodule

    ; CHECK:          module DUT
    ; CHECK-NOT:      endmodule
    ; EXTRACT:          /* This instance is elsewhere emitted as a bind statement.
    ; EXTRACT-NEXT:        MyView_companion MyView_companion (
    ; EXTRACT-NEXT:          .[[in_uint_port:[a-zA-Z0-9_]+]]                             (in_uint),
    ; EXTRACT-NEXT:          .[[in_vec_0_port:[a-zA-Z0-9_]+]]                            (in_vec_0),
    ; EXTRACT-NEXT:          .[[in_vec_1_port:[a-zA-Z0-9_]+]]                            (in_vec_1),
    ; EXTRACT-NEXT:          .[[in_multivec_0_0_port:[a-zA-Z0-9_]+]]                     (in_multivec_0_0),
    ; EXTRACT-NEXT:          .[[in_multivec_0_1_port:[a-zA-Z0-9_]+]]                     (in_multivec_0_1),
    ; EXTRACT-NEXT:          .[[in_multivec_0_2_port:[a-zA-Z0-9_]+]]                     (in_multivec_0_2),
    ; EXTRACT-NEXT:          .[[in_multivec_1_0_port:[a-zA-Z0-9_]+]]                     (in_multivec_1_0),
    ; EXTRACT-NEXT:          .[[in_multivec_1_1_port:[a-zA-Z0-9_]+]]                     (in_multivec_1_1),
    ; EXTRACT-NEXT:          .[[in_multivec_1_2_port:[a-zA-Z0-9_]+]]                     (in_multivec_1_2),
    ; EXTRACT-NEXT:          .[[in_vecOfBundle_0_sint_port:[a-zA-Z0-9_]+]]               (in_vecOfBundle_0_sint),
    ; EXTRACT-NEXT:          .[[in_vecOfBundle_0_uint_port:[a-zA-Z0-9_]+]]               (in_vecOfBundle_0_uint),
    ; EXTRACT-NEXT:          .[[in_vecOfBundle_1_sint_port:[a-zA-Z0-9_]+]]               (in_vecOfBundle_1_sint),
    ; EXTRACT-NEXT:          .[[in_vecOfBundle_1_uint_port:[a-zA-Z0-9_]+]]               (in_vecOfBundle_1_uint),
    ; EXTRACT-NEXT:          .[[in_otherOther_other_sint_port:[a-zA-Z0-9_]+]]            (in_otherOther_other_sint),
    ; EXTRACT-NEXT:          .[[in_otherOther_other_uint_port:[a-zA-Z0-9_]+]]            (in_otherOther_other_uint),
    ; EXTRACT-NEXT:          .[[DUT_submodule_in_vec_0_port:[a-zA-Z0-9_]+]]              (DUT.submodule.in_vec_0_probe),
    ; EXTRACT-NEXT:          .[[DUT_submodule_in_vec_1_port:[a-zA-Z0-9_]+]]              (DUT.submodule.in_vec_1_probe),
    ; EXTRACT-NEXT:          .[[DUT_submodule_in_vecOfBundle_0_sint_port:[a-zA-Z0-9_]+]] (DUT.submodule.in_vecOfBundle_0_sint_probe),
    ; EXTRACT-NEXT:          .[[DUT_submodule_in_vecOfBundle_0_uint_port:[a-zA-Z0-9_]+]] (DUT.submodule.in_vecOfBundle_0_uint_probe),
    ; EXTRACT-NEXT:          .[[DUT_submodule_in_vecOfBundle_1_sint_port:[a-zA-Z0-9_]+]] (DUT.submodule.in_vecOfBundle_1_sint_probe),
    ; EXTRACT-NEXT:          .[[DUT_submodule_in_vecOfBundle_1_uint_port:[a-zA-Z0-9_]+]] (DUT.submodule.in_vecOfBundle_1_uint_probe),
    ; EXTRACT-NEXT:          .[[ExtModuleWithPort_source_1_port:[a-zA-Z0-9_]+]]          ([[ExtModuleWithPort_source_1_wire:[a-zA-Z0-9_]+]])
    ; EXTRACT-NEXT:        );
    ; EXTRACT-NEXT:     */
    ; NOEXTRACT:        {{^ *}}MyView_companion MyView_companion (
    ; NOEXTRACT-NEXT:        .[[DUT_w_uint]]                               (w_uint),
    ; NOEXTRACT-NEXT:        .[[DUT_w_vec_0]]                              (w_vec_0),
    ; NOEXTRACT-NEXT:        .[[DUT_w_vec_1]]                              (w_vec_1),
    ; NOEXTRACT-NEXT:        .[[DUT_w_multivec_0_0]]                       (w_multivec_0_0),
    ; NOEXTRACT-NEXT:        .[[DUT_w_multivec_0_1]]                       (w_multivec_0_1),
    ; NOEXTRACT-NEXT:        .[[DUT_w_multivec_0_2]]                       (w_multivec_0_2),
    ; NOEXTRACT-NEXT:        .[[DUT_w_multivec_1_0]]                       (w_multivec_1_0),
    ; NOEXTRACT-NEXT:        .[[DUT_w_multivec_1_1]]                       (w_multivec_1_1),
    ; NOEXTRACT-NEXT:        .[[DUT_w_multivec_1_2]]                       (w_multivec_1_2),
    ; NOEXTRACT-NEXT:        .[[DUT_w_vecOfBundle_0_sint]]                 (w_vecOfBundle_0_sint),
    ; NOEXTRACT-NEXT:        .[[DUT_w_vecOfBundle_0_uint]]                 (w_vecOfBundle_0_uint),
    ; NOEXTRACT-NEXT:        .[[DUT_w_vecOfBundle_1_sint]]                 (w_vecOfBundle_1_sint),
    ; NOEXTRACT-NEXT:        .[[DUT_w_vecOfBundle_1_uint]]                 (w_vecOfBundle_1_uint),
    ; NOEXTRACT-NEXT:        .[[DUT_w_otherOther_other_sint]]              (w_otherOther_other_sint),
    ; NOEXTRACT-NEXT:        .[[DUT_w_otherOther_other_uint]]              (w_otherOther_other_uint),
    ; NOEXTRACT-NEXT:        .[[DUT_submodule_w_vec_0]]                    (DUT.submodule.w_vec_0_probe),
    ; NOEXTRACT-NEXT:        .[[DUT_submodule_w_vec_1]]                    (DUT.submodule.w_vec_1_probe),
    ; NOEXTRACT-NEXT:        .[[DUT_submodule_w_vecOfBundle_0_sint]]       (DUT.submodule.w_vecOfBundle_0_sint_probe),
    ; NOEXTRACT-NEXT:        .[[DUT_submodule_w_vecOfBundle_0_uint]]       (DUT.submodule.w_vecOfBundle_0_uint_probe),
    ; NOEXTRACT-NEXT:        .[[DUT_submodule_w_vecOfBundle_1_sint]]       (DUT.submodule.w_vecOfBundle_1_sint_probe),
    ; NOEXTRACT-NEXT:        .[[DUT_submodule_w_vecOfBundle_1_uint]]       (DUT.submodule.w_vecOfBundle_1_uint_probe)
    ; NOEXTRACT-NEXT:        .[[ExtModuleWithPort_source_1:[a-zA-Z0-9_]+]] ([[ExtModuleWithPort_source_1_wire:[a-zA-Z0-9_]+]])
    ; NOEXTRACT-NEXT:      );

    ; CHECK:              ExtModuleWithPort ext (
    ; CHECK-NEXT:           .source_0 (
    ; CHECK-NEXT:           .source_1 ([[ExtModuleWithPort_source_1_wire]])
    ; CHECK-NEXT:         );

    ; EXTRACT:        FILE ".{{[/\]}}BlackBox_DUT.v"
    ; EXTRACT:        FILE "Wire{{[/\]}}firrtl{{[/\]}}gct{{[/\]}}BlackBox_GCT.v"
    ; EXTRACT:        FILE ".{{[/\]}}BlackBox_DUTAndGCT.v"

    ; EXTRACT:        FILE "firrtl_black_box_resource_files.f"
    ; EXTRACT-NOT:    FILE
    ; EXTRACT:          BlackBox_DUT.v
    ; EXTRACT-NEXT:     BlackBox_DUTAndGCT.v
    ; EXTRACT-NEXT:     Wire{{[/\]}}firrtl{{[/\]}}gct{{[/\]}}BlackBox_GCT.v

    ; EXTRACT:        FILE "Wire{{[/\]}}firrtl{{[/\]}}gct{{[/\]}}MyView_companion.sv"
    ; NOEXTRACT-NOT:  FILE {{.*}}{{[/\]}}MyView_companion.sv
    ; EXTRACT:        module MyView_companion(
    ; EXTRACT:          MyInterface MyView();
    ; EXTRACT:          assign MyView.uint = [[in_uint_port]];
    ; EXTRACT-NEXT:     assign MyView.vec[0] = [[in_vec_0_port]];
    ; EXTRACT-NEXT:     assign MyView.vec[1] = [[in_vec_1_port]];
    ; EXTRACT-NEXT:     assign MyView.multivec[0][0] = [[in_multivec_0_0_port]];
    ; EXTRACT-NEXT:     assign MyView.multivec[0][1] = [[in_multivec_0_1_port]];
    ; EXTRACT-NEXT:     assign MyView.multivec[0][2] = [[in_multivec_0_2_port]];
    ; EXTRACT-NEXT:     assign MyView.multivec[1][0] = [[in_multivec_1_0_port]];
    ; EXTRACT-NEXT:     assign MyView.multivec[1][1] = [[in_multivec_1_1_port]];
    ; EXTRACT-NEXT:     assign MyView.multivec[1][2] = [[in_multivec_1_2_port]];
    ; EXTRACT-NEXT:     assign MyView.vecOfBundle[0].sint = [[in_vecOfBundle_0_sint_port]];
    ; EXTRACT-NEXT:     assign MyView.vecOfBundle[0].uint = [[in_vecOfBundle_0_uint_port]];
    ; EXTRACT-NEXT:     assign MyView.vecOfBundle[1].sint = [[in_vecOfBundle_1_sint_port]];
    ; EXTRACT-NEXT:     assign MyView.vecOfBundle[1].uint = [[in_vecOfBundle_1_uint_port]];
    ; EXTRACT-NEXT:     assign MyView.otherOther.other.sint = [[in_otherOther_other_sint_port]];
    ; EXTRACT-NEXT:     assign MyView.otherOther.other.uint = [[in_otherOther_other_uint_port]];
    ; EXTRACT-NEXT:     assign MyView.sub_uint = 1'h1;
    ; EXTRACT-NEXT:     assign MyView.sub_vec[0] = [[DUT_submodule_in_vec_0_port]];
    ; EXTRACT-NEXT:     assign MyView.sub_vec[1] = [[DUT_submodule_in_vec_1_port]];
    ; EXTRACT-NEXT:     assign MyView.sub_vecOfBundle[0].sint = [[DUT_submodule_in_vecOfBundle_0_sint_port]];
    ; EXTRACT-NEXT:     assign MyView.sub_vecOfBundle[0].uint = [[DUT_submodule_in_vecOfBundle_0_uint_port]];
    ; EXTRACT-NEXT:     assign MyView.sub_vecOfBundle[1].sint = [[DUT_submodule_in_vecOfBundle_1_sint_port]];
    ; EXTRACT-NEXT:     assign MyView.sub_vecOfBundle[1].uint = [[DUT_submodule_in_vecOfBundle_1_uint_port]];
    ; EXTRACT-NEXT:     assign MyView.ext_port_1 = [[ExtModuleWithPort_source_1_port]];
    ; EXTRACT:          Tap tap (
    ; EXTRACT-NEXT:       .b     (r),
    ; EXTRACT-NEXT:       .clock (_tap_clock),
    ; EXTRACT-NEXT:       .a     (_tap_a)
    ; EXTRACT-NEXT:     );
    ; EXTRACT-NEXT:   endmodule

    ; EXTRACT:        FILE "Wire{{[/\]}}firrtl{{[/\]}}bindings.sv"
    ; EXTRACT-NOT:    FILE
    ; EXTRACT:        bind DUT MyView_companion MyView_companion{{ *}}(
    ; EXTRACT-NEXT:          .[[in_uint_port]]                             (in_uint),
    ; EXTRACT-NEXT:          .[[in_vec_0_port]]                            (in_vec_0),
    ; EXTRACT-NEXT:          .[[in_vec_1_port]]                            (in_vec_1),
    ; EXTRACT-NEXT:          .[[in_multivec_0_0_port]]                     (in_multivec_0_0),
    ; EXTRACT-NEXT:          .[[in_multivec_0_1_port]]                     (in_multivec_0_1),
    ; EXTRACT-NEXT:          .[[in_multivec_0_2_port]]                     (in_multivec_0_2),
    ; EXTRACT-NEXT:          .[[in_multivec_1_0_port]]                     (in_multivec_1_0),
    ; EXTRACT-NEXT:          .[[in_multivec_1_1_port]]                     (in_multivec_1_1),
    ; EXTRACT-NEXT:          .[[in_multivec_1_2_port]]                     (in_multivec_1_2),
    ; EXTRACT-NEXT:          .[[in_vecOfBundle_0_sint_port]]               (in_vecOfBundle_0_sint),
    ; EXTRACT-NEXT:          .[[in_vecOfBundle_0_uint_port]]               (in_vecOfBundle_0_uint),
    ; EXTRACT-NEXT:          .[[in_vecOfBundle_1_sint_port]]               (in_vecOfBundle_1_sint),
    ; EXTRACT-NEXT:          .[[in_vecOfBundle_1_uint_port]]               (in_vecOfBundle_1_uint),
    ; EXTRACT-NEXT:          .[[in_otherOther_other_sint_port]]            (in_otherOther_other_sint),
    ; EXTRACT-NEXT:          .[[in_otherOther_other_uint_port]]            (in_otherOther_other_uint),
    ; EXTRACT-NEXT:          .[[DUT_submodule_in_vec_0_port]]              (DUT.submodule.in_vec_0_probe),
    ; EXTRACT-NEXT:          .[[DUT_submodule_in_vec_1_port]]              (DUT.submodule.in_vec_1_probe),
    ; EXTRACT-NEXT:          .[[DUT_submodule_in_vecOfBundle_0_sint_port]] (DUT.submodule.in_vecOfBundle_0_sint_probe),
    ; EXTRACT-NEXT:          .[[DUT_submodule_in_vecOfBundle_0_uint_port]] (DUT.submodule.in_vecOfBundle_0_uint_probe),
    ; EXTRACT-NEXT:          .[[DUT_submodule_in_vecOfBundle_1_sint_port]] (DUT.submodule.in_vecOfBundle_1_sint_probe),
    ; EXTRACT-NEXT:          .[[DUT_submodule_in_vecOfBundle_1_uint_port]] (DUT.submodule.in_vecOfBundle_1_uint_probe),
    ; EXTRACT-NEXT:          .[[ExtModuleWithPort_source_1_port]]          ([[ExtModuleWithPort_source_1_wire]])
    ; );
    ; NOEXTRACT-NOT:  FILE "Wire{{[/\]}}firrtl{{[/\]}}bindings.sv"

    ; EXTRACT:        FILE "Wire{{[/\]}}firrtl{{[/\]}}gct{{[/\]}}MyInterface.sv"
    ; NOEXTRACT-NOT:  FILE {{.*}}{{[/\]}}MyInterface.sv
    ; PREFIX:         interface FOO_BAR_BAZ_MyInterface;
    ; CHECK:          interface MyInterface;
    ; CHECK-NEXT:       // a wire called 'uint'
    ; CHECK-NEXT:       logic uint;
    ; CHECK-NEXT:       // a vector called 'vec'
    ; CHECK-NEXT:       logic vec[0:1];
    ; CHECK-NEXT:       // a 2D vector called 'multivec'
    ; CHECK-NEXT:       logic multivec[0:1][0:2];
    ; CHECK-NEXT:       // a vector of a bundle
    ; CHECK-NEXT:       VecOfBundle vecOfBundle[2]();
    ; CHECK-NEXT:       // another bundle
    ; CHECK-NEXT:       OtherOther otherOther();
    ; CHECK-NEXT:       // a wire called 'uint' in the submodule
    ; CHECK-NEXT:       logic sub_uint;
    ; CHECK-NEXT:       // a vector called 'vec' in the submodule
    ; CHECK-NEXT:       logic sub_vec[0:1];
    ; CHECK-NEXT:       // a vector of a bundle in the submodule with a
    ; CHECK-NEXT:       // multiline comment
    ; CHECK-NEXT:       Sub_vecOfBundle sub_vecOfBundle[2]();
    ; CHECK-NEXT:       // The second element of an external port
    ; CHECK-NEXT:       logic ext_port_1;
    ; CHECK-NEXT:       // <unsupported string type> foo_string;
    ; CHECK-NEXT:       // <unsupported integer type> foo_integer;
    ; CHECK-NEXT:       // <unsupported double type> foo_double;
    ; CHECK-NEXT:       // <unsupported boolean type> foo_boolean;
    ; CHECK-NEXT:     endinterface

    ; EXTRACT:        FILE "Wire{{[/\]}}firrtl{{[/\]}}gct{{[/\]}}VecOfBundle.sv"
    ; NOEXTRACT-NOT:  FILE {{.*}}{{[/\]}}VecOfBundle.sv
    ; PREFIX:         interface FOO_BAR_BAZ_VecOfBundle;
    ; CHECK:          interface VecOfBundle;
    ; CHECK-NEXT:       logic [1:0] sint;
    ; CHECK-NEXT:       logic [3:0] uint;
    ; CHECK-NEXT:     endinterface

    ; EXTRACT:        FILE "Wire{{[/\]}}firrtl{{[/\]}}gct{{[/\]}}OtherOther.sv"
    ; NOEXTRACT-NOT:  FILE {{.*}}{{[/\]}}OtherOther.sv
    ; PREFIX:         interface FOO_BAR_BAZ_OtherOther;
    ; CHECK:          interface OtherOther;
    ; CHECK-NEXT:       Other other();
    ; CHECK:          endinterface

    ; EXTRACT:        FILE "Wire{{[/\]}}firrtl{{[/\]}}gct{{[/\]}}Other.sv"
    ; NOEXTRACT-NOT:  FILE {{.*}}{{[/\]}}Other.sv
    ; PREFIX:         interface FOO_BAR_BAZ_Other;
    ; CHECK:          interface Other;
    ; CHECK-NEXT:       logic [1:0] sint;
    ; CHECK-NEXT:       logic [3:0] uint;
    ; CHECK:          endinterface

    ; EXTRACT:        FILE "Wire{{[/\]}}firrtl{{[/\]}}gct{{[/\]}}Sub_vecOfBundle.sv"
    ; NOEXTRACT-NOT:  FILE {{.*}}{{[/\]}}Sub_vecOfBundle.sv
    ; PREFIX:         interface FOO_BAR_BAZ_Sub_vecOfBundle;
    ; CHECK:          interface Sub_vecOfBundle;
    ; CHECK-NEXT:       logic [1:0] sint;
    ; CHECK-NEXT:       logic [3:0] uint;
    ; CHECK:          endinterface

    ; YAML:           FILE "Wire{{[/\]}}firrtl{{[/\]}}gct{{[/\]}}gct.yaml"
    ; YAML:             - name: MyInterface
    ; YAML-NEXT:          fields:
    ; YAML-NEXT:            - name: uint
    ; YAML-NEXT:              description: 'a wire called ''uint'''
    ; YAML-NEXT:              dimensions: [  ]
    ; YAML-NEXT:              width: 1
    ; YAML-NEXT:            - name: vec
    ; YAML-NEXT:              description: 'a vector called ''vec'''
    ; YAML-NEXT:              dimensions: [ 2 ]
    ; YAML-NEXT:              width: 1
    ; YAML-NEXT:            - name: multivec
    ; YAML-NEXT:              description: 'a 2D vector called ''multivec'''
    ; YAML-NEXT:              dimensions: [ 3, 2 ]
    ; YAML-NEXT:              width: 1
    ; YAML-NEXT:            - name: sub_uint
    ; YAML-NEXT:              description: 'a wire called ''uint'' in the submodule'
    ; YAML-NEXT:              dimensions: [  ]
    ; YAML-NEXT:              width: 1
    ; YAML-NEXT:            - name: sub_vec
    ; YAML-NEXT:              description: 'a vector called ''vec'' in the submodule'
    ; YAML-NEXT:              dimensions: [ 2 ]
    ; YAML-NEXT:              width: 1
    ; YAML-NEXT:            - name: ext_port_1
    ; YAML-NEXT:              description: The second element of an external port
    ; YAML-NEXT:              dimensions: [ ]
    ; YAML-NEXT:              width: 1
    ; YAML-NEXT:          instances:
    ; YAML-NEXT:            - name: vecOfBundle
    ; YAML-NEXT:              description: a vector of a bundle
    ; YAML-NEXT:              dimensions: [ 2 ]
    ; YAML-NEXT:              interface:
    ; YAML-NEXT:                name: VecOfBundle
    ; YAML-NEXT:                fields:
    ; YAML-NEXT:                  - name: sint
    ; YAML-NEXT:                    dimensions: [  ]
    ; YAML-NEXT:                    width: 2
    ; YAML-NEXT:                  - name: uint
    ; YAML-NEXT:                    dimensions: [  ]
    ; YAML-NEXT:                    width: 4
    ; YAML-NEXT:                instances: []
    ; YAML-NEXT:            - name: otherOther
    ; YAML-NEXT:              description: another bundle
    ; YAML-NEXT:              dimensions: [  ]
    ; YAML-NEXT:              interface:
    ; YAML-NEXT:                name: OtherOther
    ; YAML-NEXT:                fields: []
    ; YAML-NEXT:                instances:
    ; YAML-NEXT:                  - name: other
    ; YAML-NEXT:                    dimensions: [  ]
    ; YAML-NEXT:                    interface:
    ; YAML-NEXT:                      name: Other
    ; YAML-NEXT:                      fields:
    ; YAML-NEXT:                        - name: sint
    ; YAML-NEXT:                          dimensions: [  ]
    ; YAML-NEXT:                          width: 2
    ; YAML-NEXT:                        - name: uint
    ; YAML-NEXT:                          dimensions: [  ]
    ; YAML-NEXT:                          width: 4
    ; YAML-NEXT:                      instances: []
    ; YAML-NEXT:            - name: sub_vecOfBundle
    ; YAML-NEXT:              description: "a vector of a bundle in the submodule with a\nmultiline comment"
    ; YAML-NEXT:              dimensions: [ 2 ]
    ; YAML-NEXT:              interface:
    ; YAML-NEXT:                name: Sub_vecOfBundle
    ; YAML-NEXT:                fields:
    ; YAML-NEXT:                  - name: sint
    ; YAML-NEXT:                    dimensions: [  ]
    ; YAML-NEXT:                    width: 2
    ; YAML-NEXT:                  - name: uint
    ; YAML-NEXT:                    dimensions: [  ]
    ; YAML-NEXT:                    width: 4
    ; YAML-NEXT:                instances: []
