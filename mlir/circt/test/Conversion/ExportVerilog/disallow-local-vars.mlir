// RUN: circt-opt --export-verilog %s | FileCheck %s
// RUN: circt-opt --test-apply-lowering-options='options=disallowLocalVariables' --export-verilog %s | FileCheck %s --check-prefix=DISALLOW -strict-whitespace

// This checks ExportVerilog's support for "disallowLocalVariables" which
// prevents emitting 'automatic logic' and other local declarations.

// CHECK-LABEL: module side_effect_expr
// DISALLOW-LABEL: module side_effect_expr
hw.module @side_effect_expr(%clock: i1) -> (a: i1, a2: i1) {

  // CHECK: `ifdef FOO_MACRO
  // DISALLOW: `ifdef FOO_MACRO
  sv.ifdef "FOO_MACRO" {
    // DISALLOW: logic logicOp;
    // DISALLOW: {{^    }}reg   [[SE_REG:[_A-Za-z0-9]+]];

    // CHECK:    always @(posedge clock)
    // DISALLOW: always @(posedge clock)
    sv.always posedge %clock  {
      %0 = sv.verbatim.expr "INLINE_OK" : () -> i1
      // CHECK: automatic logic logicOp;
      %logicOp = sv.logic : !hw.inout<i1>

      // This shouldn't be pushed into a reg.
      // CHECK: if (INLINE_OK)
      // DISALLOW: if (INLINE_OK)
      sv.if %0  {
        sv.fatal 1
      }

      // This should go through a reg when in "disallow" mode.
      // CHECK: if (SIDE_EFFECT)
      // DISALLOW: [[SE_REG]] = SIDE_EFFECT;
      // DISALLOW: if ([[SE_REG]])
      %1 = sv.verbatim.expr.se "SIDE_EFFECT" : () -> i1
      sv.if %1  {
        sv.fatal 1
      }
    }
  }
  // CHECK: `endif
  // DISALLOW: `endif

  // Top level things should go unmodified.
  %2 = sv.verbatim.expr "NO_SE" : () -> i1
  %3 = sv.verbatim.expr.se "YES_SE" : () -> i1

  // CHECK: assign a = NO_SE;
  // CHECK: assign a2 = YES_SE;
  // DISALLOW: assign a = NO_SE;
  // DISALLOW: assign a2 = YES_SE;
  hw.output %2, %3: i1, i1
}

// CHECK-LABEL: module hoist_expressions
// DISALLOW-LABEL: module hoist_expressions
hw.module @hoist_expressions(%clock: i1, %x: i8, %y: i8, %z: i8) {
  // DISALLOW: wire [7:0] [[ADD:[_A-Za-z0-9]+]] = x + y;

  %fd = hw.constant 0x80000002 : i32

  // CHECK:    always @(posedge clock)
  // DISALLOW: always @(posedge clock)
  sv.always posedge %clock  {
    %0 = comb.add %x, %y: i8
    %1 = comb.icmp eq %0, %z : i8

    // This shouldn't be touched.
    // CHECK: if (_GEN == z) begin
    // DISALLOW: if (_GEN == z) begin
    sv.if %1  {
      // CHECK: $fwrite(32'h80000002, "Hi %x\n", _GEN * z);
      // DISALLOW: $fwrite(32'h80000002, "Hi %x\n", _GEN * z);
      %2 = comb.mul %0, %z : i8
      sv.fwrite %fd, "Hi %x\0A"(%2) : i8
      sv.fatal 1
    }
  }

  // Check out wires.
  // CHECK: wire [7:0] myWire = x;
  // DISALLOW: wire [7:0] myWire = x;
  %myWire = sv.wire : !hw.inout<i8>
  sv.assign %myWire, %x : i8

  // CHECK: always @(posedge clock)
  // DISALLOW: always @(posedge clock)
  sv.always posedge %clock  {
    %wireout = sv.read_inout %myWire : !hw.inout<i8>
    %3 = comb.add %x, %wireout: i8
    %4 = comb.icmp eq %3, %z : i8
    // CHECK: if (x + myWire == z)
    // DISALLOW: if (x + myWire == z)
    sv.if %4  {
      sv.fatal 1
    }
 }

  hw.output
}

// CHECK-LABEL: module always_inline_expr
// DISALLOW-LABEL: module always_inline_expr
// https://github.com/llvm/circt/issues/1705
hw.module @always_inline_expr(%ro_clock_0: i1, %ro_en_0: i1, %ro_addr_0: i1, %wo_clock_0: i1, %wo_en_0: i1, %wo_addr_0: i1, %wo_mask_0: i1, %wo_data_0: i5) -> (ro_data_0: i5) {
  %Memory = sv.reg  : !hw.inout<uarray<2xi5>>
  %0 = sv.array_index_inout %Memory[%ro_addr_0] : !hw.inout<uarray<2xi5>>, i1
  %1 = sv.read_inout %0 : !hw.inout<i5>
  %x_i5 = sv.constantX : i5
  %2 = comb.mux %ro_en_0, %1, %x_i5 : i5
  sv.alwaysff(posedge %wo_clock_0)  {
    %3 = comb.and %wo_en_0, %wo_mask_0 : i1
    // CHECK: if (wo_en_0 & wo_mask_0)
    // DISALLOW: if (wo_en_0 & wo_mask_0)
    sv.if %3  {
      // CHECK: Memory[wo_addr_0] <= wo_data_0;
      // DISALLOW: Memory[wo_addr_0] <= wo_data_0;
      %4 = sv.array_index_inout %Memory[%wo_addr_0] : !hw.inout<uarray<2xi5>>, i1
      sv.passign %4, %wo_data_0 : i5
    }
  }
  hw.output %2 : i5
}

// CHECK-LABEL: module EmittedDespiteDisallowed
// DISALLOW-LABEL: module EmittedDespiteDisallowed
// https://github.com/llvm/circt/issues/2216
hw.module @EmittedDespiteDisallowed(%clock: i1, %reset: i1) {
  %tick_value_2 = sv.reg  : !hw.inout<i1>
  %counter_value = sv.reg  : !hw.inout<i1>

  // Temporary reg gets introduced.
  // DISALLOW: reg [1:0] [[TEMP:.+]];

  // DISALLOW: initial begin
  sv.initial {
    // CHECK: automatic logic [1:0] _magic = magic;
    // DISALLOW: _GEN = magic;
    %RANDOM = sv.verbatim.expr.se "magic" : () -> i2 {symbols = []}

    // CHECK: tick_value_2 = _magic[0];
    // DISALLOW-NEXT: tick_value_2 = [[TEMP]][0];
    %1 = comb.extract %RANDOM from 0 : (i2) -> i1
    sv.bpassign %tick_value_2, %1 : i1

    // CHECK: counter_value = _magic[1];
    // DISALLOW-NEXT: counter_value = [[TEMP]][1];
    %2 = comb.extract %RANDOM from 1 : (i2) -> i1
    sv.bpassign %counter_value, %2 : i1
  }
  hw.output
}

// CHECK-LABEL: module ReadInoutAggregate(
hw.module @ReadInoutAggregate(%clock: i1) {
  %register = sv.reg  : !hw.inout<array<1xstruct<a: i32>>>
  sv.always posedge %clock  {
    %c0_i16 = hw.constant 0 : i16
    %false = hw.constant false
    %0 = sv.array_index_inout %register[%false] : !hw.inout<array<1xstruct<a: i32>>>, i1
    %1 = sv.struct_field_inout %0["a"] : !hw.inout<struct<a: i32>>
    %2 = sv.read_inout %1 : !hw.inout<i32>
    %3 = comb.extract %2 from 0 : (i32) -> i16
    %4 = comb.concat %c0_i16, %3 : i16, i16
    sv.passign %1, %4 : i32
  }
  // DISALLOW: always @(
  // DISALLOW-NEXT:  register[1'h0].a <= {16'h0, register[1'h0].a[15:0]};
  hw.output
}

// CHECK-LABEL: DefinedInDifferentBlock
// CHECK: `ifdef DEF
// CHECK-NEXT: initial begin
// CHECK-NEXT:   if (a == b)
// CHECK-NEXT:     $error("error")
// DISALLOW: `ifdef DEF
// DISALLOW-NEXT: initial begin
// DISALLOW-NEXT:   if (a == b)
// DISALLOW-NEXT:     $error("error")

hw.module @DefinedInDifferentBlock(%a: i1, %b: i1) {
  sv.ifdef "DEF" {
    %0 = comb.icmp eq %a, %b : i1
    sv.initial {
      sv.if %0 {
        sv.error "error"
      }
    }
  }
  hw.output
}

// CHECK-LABEL: module TemporaryWireAtDifferentBlock(
// DISALLOW-LABEL: module TemporaryWireAtDifferentBlock(
hw.module @TemporaryWireAtDifferentBlock(%a: i1) -> (b: i1) {
  // Check that %0 and %1 are not inlined.
  // CHECK:      wire [[GEN1:.+]];
  // CHECK:      wire [[GEN2:.+]] = [[GEN1]] + [[GEN1]];
  // CHECK:      if ([[GEN1]])
  // DISALLOW:   wire [[GEN1:.+]];
  // DISALLOW:   wire [[GEN2:.+]] = [[GEN1]] + [[GEN1]];
  // DISALLOW:   if ([[GEN1]])
  %1 = comb.add %0, %0 : i1
  sv.initial {
    sv.if %0 {
      sv.error "error"
    }
  }
  %0 = comb.shl %a, %a : i1
  %2 = comb.add %1, %1 : i1
  hw.output %2 : i1
}

// CHECK-LABEL: module AggregateInline(
// DISALLOW-LABEL: module AggregateInline(
hw.module @AggregateInline(%clock: i1) {
  %c0_i16 = hw.constant 0 : i16
  %false = hw.constant false
  // CHECK: wire [15:0]{{ *}}[[GEN:.+]];
  // DISALLOW: wire [15:0]{{ *}}[[GEN:.+]];

  %register = sv.reg  : !hw.inout<struct<a: i32>>
  sv.always posedge %clock  {
    // %4 can not be inlined because %3 uses %2.
    %4 = comb.concat %c0_i16, %3 : i16, i16
    // DISALLOW: register.a <= {16'h0, [[GEN]]};
    // CHECK: register.a <= {16'h0, [[GEN]]};
    sv.passign %1, %4 : i32
  }
  %1 = sv.struct_field_inout %register["a"] : !hw.inout<struct<a: i32>>
  %2 = sv.read_inout %1 : !hw.inout<i32>
  %3 = comb.extract %2 from 0 : (i32) -> i16
  // DISALLOW: assign [[GEN]] = register.a[15:0]
  // CHECK: assign [[GEN]] = register.a[15:0]
  hw.output
}
