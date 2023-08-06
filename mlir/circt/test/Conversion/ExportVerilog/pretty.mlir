// RUN: circt-opt -export-verilog --split-input-file %s | FileCheck %s --strict-whitespace --match-full-lines

sv.interface @IValidReady_Struct  {
  sv.interface.signal @data : !hw.struct<foo: !hw.array<72xi1>, bar: !hw.array<128xi1>, baz: !hw.array<224xi1>>
}

// CHECK-LABEL:module structs({{.*}}
//      CHECK:  assign _GEN =
// CHECK-NEXT:    '{foo: ({_GEN_1, _GEN_0}),
// CHECK-NEXT:      bar: ({_GEN_0, _GEN_0}),
// CHECK-NEXT:      baz:
// CHECK-NEXT:        ({_GEN_1,
// CHECK-NEXT:          _GEN_1,
// CHECK-NEXT:          _GEN_1,
// CHECK-NEXT:          _GEN_1,
// CHECK-NEXT:          _GEN_1,
// CHECK-NEXT:          _GEN_1,
// CHECK-NEXT:          _GEN_1,
// CHECK-NEXT:          _GEN_1,
// CHECK-NEXT:          _GEN_1,
// CHECK-NEXT:          _GEN_1,
// CHECK-NEXT:          _GEN_1,
// CHECK-NEXT:          _GEN_1,
// CHECK-NEXT:          _GEN_0,
// CHECK-NEXT:          _GEN_0})};{{.*}}
hw.module @structs(%clk: i1, %rstn: i1) {
  %0 = sv.interface.instance name "iface" : !sv.interface<@IValidReady_Struct>
  sv.interface.signal.assign %0(@IValidReady_Struct::@data) = %s : !hw.struct<foo: !hw.array<72xi1>, bar: !hw.array<128xi1>, baz: !hw.array<224xi1>>
  %c0 = hw.constant 0 : i8
  %c64 = hw.constant 100000 : i64
  %16 = hw.bitcast %c64 : (i64) -> !hw.array<64xi1>
  %58 = hw.bitcast %c0 : (i8) -> !hw.array<8xi1>
  %90 = hw.array_concat %58, %16 : !hw.array<8xi1>, !hw.array<64xi1>
  %91 = hw.array_concat %16, %16 : !hw.array<64xi1>, !hw.array<64xi1>
  %92 = hw.array_concat %58, %58, %58, %58, %58, %58, %58, %58, %58, %58, %58, %58, %16, %16 : !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<64xi1>, !hw.array<64xi1>
  %s = hw.struct_create (%90, %91, %92) : !hw.struct<foo: !hw.array<72xi1>, bar: !hw.array<128xi1>, baz: !hw.array<224xi1>>
}

// -----

// CHECK-LABEL:module CoverAssert({{.*}}
hw.module @CoverAssert(
  %clock: i1, %reset: i1,
  %eeeeee_fffff_gggggg_hhh_i_jjjjj_kkkkkkkkk_lllllll_mmmmmmmmm_nnnnnnnn_0: i4) {
    %c0_i4 = hw.constant 0 : i4
    %true = hw.constant true

    %0 = comb.icmp eq %eeeeee_fffff_gggggg_hhh_i_jjjjj_kkkkkkkkk_lllllll_mmmmmmmmm_nnnnnnnn_0, %c0_i4 : i4
    %1 = comb.icmp eq %eeeeee_fffff_gggggg_hhh_i_jjjjj_kkkkkkkkk_lllllll_mmmmmmmmm_nnnnnnnn_0, %c0_i4 : i4

    %2 = comb.xor bin %reset, %true : i1
    %3 = comb.xor bin %reset, %true : i1
    %4 = comb.and bin %0, %2 : i1
    %5 = comb.and bin %1, %3 : i1

//      CHECK:  cover__information_label:
// CHECK-NEXT:    cover property (@(posedge clock)
// CHECK-NEXT:                    eeeeee_fffff_gggggg_hhh_i_jjjjj_kkkkkkkkk_lllllll_mmmmmmmmm_nnnnnnnn_0 == 4'h0
// CHECK-NEXT:                    & ~reset);{{.*}}
    sv.cover.concurrent posedge %clock, %4 label "cover__information_label"

//      CHECK:  assert__label:
// CHECK-NEXT:    assert property (@(posedge clock)
// CHECK-NEXT:                     eeeeee_fffff_gggggg_hhh_i_jjjjj_kkkkkkkkk_lllllll_mmmmmmmmm_nnnnnnnn_0 == 4'h0
// CHECK-NEXT:                     & ~reset)
// CHECK-NEXT:    else $error("assert failed");	{{.*}}
    sv.assert.concurrent posedge %clock, %5 label "assert__label" message "assert failed"
}

hw.module @MuxChain(%a_0: i1, %a_1: i1, %a_2: i1, %c_0: i1, %c_1: i1, %c_2: i1) -> (out: i1) {
  %0 = comb.mux bin %a_1, %c_1, %c_0 : i1
  %1 = comb.mux bin %a_0, %0, %c_2 : i1
  %2 = comb.mux bin %a_2, %1, %c_1 : i1
  %3 = comb.mux bin %a_1, %c_0, %2 : i1
  %4 = comb.mux bin %a_0, %c_2, %3 : i1
  %5 = comb.mux bin %a_2, %c_1, %4 : i1
  %6 = comb.mux bin %a_1, %c_0, %5 : i1
  %7 = comb.mux bin %a_0, %c_2, %6 : i1
  %8 = comb.mux bin %a_2, %c_1, %7 : i1
  %9 = comb.mux bin %a_1, %c_0, %8 : i1
  %10 = comb.mux bin %a_0, %c_2, %9 : i1
  %11 = comb.mux bin %a_2, %c_1, %10 : i1
  %12 = comb.mux bin %a_1, %c_0, %11 : i1
  %13 = comb.mux bin %a_0, %c_2, %12 : i1
  %14 = comb.mux bin %a_2, %c_1, %13 : i1
  %15 = comb.mux bin %a_1, %c_0, %14 : i1
  %16 = comb.mux bin %a_0, %c_2, %15 : i1
  %17 = comb.mux bin %a_2, %c_1, %16 : i1
  %18 = comb.mux bin %a_1, %c_0, %17 : i1
  %19 = comb.mux bin %a_0, %c_2, %18 : i1
  %20 = comb.mux bin %a_2, %c_1, %19 : i1
  %21 = comb.mux bin %a_1, %c_0, %20 : i1
  %22 = comb.mux bin %a_0, %c_2, %21 : i1
  %23 = comb.mux bin %a_2, %c_1, %22 : i1
  %24 = comb.mux bin %a_1, %c_0, %23 : i1
  %25 = comb.mux bin %a_0, %c_2, %24 : i1
  %26 = comb.mux bin %a_0, %c_1, %25 : i1
  %27 = comb.mux bin %a_0, %c_0, %26 : i1
  hw.output %27 : i1

//      CHECK:  assign out =
// CHECK-NEXT:    a_0
// CHECK-NEXT:      ? c_0
// CHECK-NEXT:      : a_0
// CHECK-NEXT:          ? c_1
// CHECK-NEXT:          : a_0
// CHECK-NEXT:              ? c_2
// CHECK-NEXT:              : a_1
// CHECK-NEXT:                  ? c_0
// CHECK-NEXT:                  : a_2
// CHECK-NEXT:                      ? c_1
// CHECK-NEXT:                      : a_0
// CHECK-NEXT:                          ? c_2
// CHECK-NEXT:                          : a_1
// CHECK-NEXT:                              ? c_0
// CHECK-NEXT:                              : a_2
// CHECK-NEXT:                                  ? c_1
// CHECK-NEXT:                                  : a_0
// CHECK-NEXT:                                      ? c_2
// CHECK-NEXT:                                      : a_1
// CHECK-NEXT:                                          ? c_0
// CHECK-NEXT:                                          : a_2
// CHECK-NEXT:                                              ? c_1
// CHECK-NEXT:                                              : a_0
// CHECK-NEXT:                                                  ? c_2
// CHECK-NEXT:                                                  : a_1
// CHECK-NEXT:                                                      ? c_0
// CHECK-NEXT:                                                      : a_2
// CHECK-NEXT:                                                          ? c_1
// CHECK-NEXT:                                                          : a_0
// CHECK-NEXT:                                                              ? c_2
// CHECK-NEXT:                                                              : a_1
// CHECK-NEXT:                                                                  ? c_0
// CHECK-NEXT:                                                                  : a_2
// CHECK-NEXT:                                                                      ? c_1
// CHECK-NEXT:                                                                      : a_0
// CHECK-NEXT:                                                                          ? c_2
// CHECK-NEXT:                                                                          : a_1
// CHECK-NEXT:                                                                              ? c_0
// CHECK-NEXT:                                                                              : a_2
// CHECK-NEXT:                                                                                  ? c_1
// CHECK-NEXT:                                                                                  : a_0
// CHECK-NEXT:                                                                                      ? c_2
// CHECK-NEXT:                                                                                      : a_1
//            ------------------------------------------------------------------------------------------v (margin=90)
// CHECK-NEXT:                                                                                          ? c_0
// CHECK-NEXT:                                                                                          : a_2
// CHECK-NEXT:                                                                                              ? c_1
// CHECK-NEXT:                                                                                              : a_0
// CHECK-NEXT:                                                                                                  ? c_2
// CHECK-NEXT:                                                                                                  : a_1
// CHECK-NEXT:                                                                                                      ? c_0
// CHECK-NEXT:                                                                                                      : a_2
// CHECK-NEXT:                                                                                                          ? (a_0
// CHECK-NEXT:                                                                                                               ? (a_1
// CHECK-NEXT:                                                                                                                    ? c_1
// CHECK-NEXT:                                                                                                                    : c_0)
// CHECK-NEXT:                                                                                                               : c_2)
// CHECK-NEXT:                                                                                                          : c_1;{{.*}}
}

// -----

// CHECK-LABEL:module svattrs{{.*}}
hw.module @svattrs() {
//      CHECK:  (* dont_merge, dont_retime = true, foo0 = bar0, foo1 = bar1, foo2 = bar2, foo3 = bar3,
// CHECK-NEXT:     foo4 = bar4, foo5 = bar5, foo6 = bar6 *)
// CHECK-NEXT:  reg [9:0] reg0;{{.*}}
  %reg0 = sv.reg {
    sv.attributes = [
      #sv.attribute<"dont_merge">,
      #sv.attribute<"dont_retime" ="true">,
      #sv.attribute<"foo0"="bar0">,
      #sv.attribute<"foo1"="bar1">,
      #sv.attribute<"foo2"="bar2">,
      #sv.attribute<"foo3"="bar3">,
      #sv.attribute<"foo4"="bar4">,
      #sv.attribute<"foo5"="bar5">,
      #sv.attribute<"foo6"="bar6">
   ]} : !hw.inout<i10>

//      CHECK:  (* start *)
// CHECK-NEXT:  /* foo0 = bar0, foo1 = bar1, foo2 = bar2, foo3 = bar3, foo4 = bar4, foo5 = bar5,
// CHECK-NEXT:     foo6 = bar6 */
// CHECK-NEXT:  (* foo0 = bar0, foo1 = bar1, foo2 = bar2, foo3 = bar3, foo4 = bar4, foo5 = bar5,
// CHECK-NEXT:     foo6 = bar6 *)
// CHECK-NEXT:  reg [9:0] reg1;{{.*}}
  %reg1 = sv.reg {
    sv.attributes = [
      #sv.attribute<"start">,
      #sv.attribute<"foo0"="bar0", emitAsComment>,
      #sv.attribute<"foo1"="bar1", emitAsComment>,
      #sv.attribute<"foo2"="bar2", emitAsComment>,
      #sv.attribute<"foo3"="bar3", emitAsComment>,
      #sv.attribute<"foo4"="bar4", emitAsComment>,
      #sv.attribute<"foo5"="bar5", emitAsComment>,
      #sv.attribute<"foo6"="bar6", emitAsComment>,
      #sv.attribute<"foo0"="bar0">,
      #sv.attribute<"foo1"="bar1">,
      #sv.attribute<"foo2"="bar2">,
      #sv.attribute<"foo3"="bar3">,
      #sv.attribute<"foo4"="bar4">,
      #sv.attribute<"foo5"="bar5">,
      #sv.attribute<"foo6"="bar6">
   ]} : !hw.inout<i10>

// Put containers on same line if they fit!
//      CHECK:  (* start *) /* comment */ (* end *)
// CHECK-NEXT:  reg [9:0] reg2;{{.*}}
  %reg2 = sv.reg {
    sv.attributes = [
      #sv.attribute<"start">,
      #sv.attribute<"comment", emitAsComment>,
      #sv.attribute<"end">
   ]} : !hw.inout<i10>

// Check behavior where some fit and some don't:
// (notably don't glue '(* end *)' after the comment container)
//      CHECK:  (* start *)
// CHECK-NEXT:  /* foo0 = bar0, foo1 = bar1, foo2 = bar2, foo3 = bar3, foo4 = bar4, foo5 = bar5,
// CHECK-NEXT:     foo6 = bar6 */
// CHECK-NEXT:  (* end *)
// CHECK-NEXT:  reg [9:0] reg3;{{.*}}
  %reg3 = sv.reg {
    sv.attributes = [
      #sv.attribute<"start">,
      #sv.attribute<"foo0"="bar0", emitAsComment>,
      #sv.attribute<"foo1"="bar1", emitAsComment>,
      #sv.attribute<"foo2"="bar2", emitAsComment>,
      #sv.attribute<"foo3"="bar3", emitAsComment>,
      #sv.attribute<"foo4"="bar4", emitAsComment>,
      #sv.attribute<"foo5"="bar5", emitAsComment>,
      #sv.attribute<"foo6"="bar6", emitAsComment>,
      #sv.attribute<"end">
   ]} : !hw.inout<i10>
}

// -----

sv.macro.decl @RANDOM

// CHECK-LABEL:module ForStatement{{.*}}
hw.module @ForStatement(%aaaaaaaaaaa: i5, %xxxxxxxxxxxxxxx : i2, %yyyyyyyyyyyyyyy : i2, %zzzzzzzzzzzzzzz : i2) -> () {
  %_RANDOM = sv.logic : !hw.inout<uarray<3xi32>>
  sv.initial {
    %x_and_y = comb.and %xxxxxxxxxxxxxxx, %yyyyyyyyyyyyyyy : i2
    %x_or_y = comb.or %xxxxxxxxxxxxxxx, %yyyyyyyyyyyyyyy : i2
    %lowerBound = comb.sub %x_and_y, %x_or_y : i2
    %upperBound = comb.and %x_or_y, %x_and_y : i2
    %eq = comb.icmp eq %xxxxxxxxxxxxxxx, %yyyyyyyyyyyyyyy : i2
    %step = comb.mux %eq, %x_and_y, %x_or_y : i2
    //      CHECK:    for (logic [1:0] iiiiiiiiiiiiiiiiiiiiiiiii = _GEN - _GEN_0;
    // CHECK-NEXT:         iiiiiiiiiiiiiiiiiiiiiiiii < _GEN_0 & _GEN;
    // CHECK-NEXT:         iiiiiiiiiiiiiiiiiiiiiiiii +=
    // CHECK-NEXT:           xxxxxxxxxxxxxxx == yyyyyyyyyyyyyyy ? _GEN : _GEN_0) begin
    // CHECK-NEXT:      _RANDOM[iiiiiiiiiiiiiiiiiiiiiiiii] = `RANDOM;{{.*}}
    // CHECK-NEXT:    end{{.*}}
    sv.for %iiiiiiiiiiiiiiiiiiiiiiiii = %lowerBound to %upperBound step %step : i2 {
      %RANDOM = sv.macro.ref.se @RANDOM() : () -> i32
      %index = sv.array_index_inout %_RANDOM[%iiiiiiiiiiiiiiiiiiiiiiiii] : !hw.inout<uarray<3xi32>>, i2
      sv.bpassign %index, %RANDOM : i32
    }
  }
}

// -----

sv.macro.decl @TEST_COND

// CHECK-LABEL:module TestCond{{.*}}
// CHECK-NEXT:  `ifdef TEST_COND_{{.*}}
// CHECK-NEXT:    `define TEST_COND TEST_COND_
// CHECK-NEXT:  `else  // TEST_COND_
// CHECK-NEXT:    `define TEST_COND 1
// CHECK-NEXT:  `endif // TEST_COND_
hw.module @TestCond() {
  sv.ifdef "TEST_COND_" {
   sv.macro.def @TEST_COND "TEST_COND_"
  } else {
   sv.macro.def @TEST_COND "1"
  }
  hw.output
}
