// RUN: circt-opt %s -test-apply-lowering-options='options=explicitBitcast,maximumNumberOfTermsPerExpression=10,emitBindComments' -export-verilog -verify-diagnostics | FileCheck %s

// CHECK-LABEL: module M1
// CHECK-NEXT:    #(parameter [41:0] param1) (
hw.module @M1<param1: i42>(%clock : i1, %cond : i1, %val : i8) {
  %wire42 = sv.reg : !hw.inout<i42>
  %forceWire = sv.wire sym @wire1 : !hw.inout<i1>
  %partSelectReg = sv.reg : !hw.inout<i42>

  %fd = hw.constant 0x80000002 : i32

  %c11_i42 = hw.constant 11: i42
  // CHECK: localparam [41:0]{{ *}} param_x = 42'd11;
  %param_x = sv.localparam { value = 11 : i42 } : i42

  // CHECK: localparam [41:0]{{ *}} param_y = param1;
  %param_y = sv.localparam { value = #hw.param.decl.ref<"param1"> : i42 } : i42

  // CHECK:        logic{{ *}} [7:0]{{ *}} logic_op = val;
  // CHECK-NEXT: struct packed {logic b; } logic_op_struct;
  %logic_op = sv.logic : !hw.inout<i8>
  %logic_op_struct = sv.logic : !hw.inout<struct<b: i1>>
  sv.assign %logic_op, %val: i8

  // CHECK:           (* sv attr *)
  // CHECK-NEXT:      always @(posedge clock) begin
  sv.always posedge %clock {
    // CHECK: automatic logic [7:0]                     logic_op_procedural = val;
    // CHECK-NEXT: automatic       struct packed {logic b; } logic_op_struct_procedural

    // CHECK: force forceWire = cond;
    sv.force %forceWire, %cond : i1
    %logic_op_procedural = sv.logic : !hw.inout<i8>
    %logic_op_struct_procedural = sv.logic : !hw.inout<struct<b: i1>>

    sv.bpassign %logic_op_procedural, %val: i8
  // CHECK-NEXT:   `ifndef SYNTHESIS
    sv.ifdef.procedural "SYNTHESIS" {
    } else {
  // CHECK-NEXT:     if ((`PRINTF_COND_) & 1'bx & 1'bz & 1'bz & cond & forceWire)
      %tmp = sv.macro.ref @PRINTF_COND_() : () -> i1
      %verb_tmp = sv.verbatim.expr "{{0}}" : () -> i1 {symbols = [#hw.innerNameRef<@M1::@wire1>] }
      %tmp1 = sv.constantX : i1
      %tmp2 = sv.constantZ : i1
      %tmp3 = comb.and %tmp, %tmp1, %tmp2, %tmp2, %cond, %verb_tmp : i1
      sv.if %tmp3 {
  // CHECK-NEXT:       $fwrite(32'h80000002, "Hi\n");
        sv.fwrite %fd, "Hi\n"
      }

  // CHECK-NEXT: release forceWire;
    sv.release %forceWire : !hw.inout<i1>
  // CHECK-NEXT:   `endif
  // CHECK-NEXT: end // always @(posedge)
    }
  } {sv.attributes = [#sv.attribute<"sv attr">]}

  // CHECK-NEXT: always @(negedge clock) begin
  // CHECK-NEXT: end // always @(negedge)
  sv.always negedge %clock {
  }

  // CHECK-NEXT: always @(edge clock) begin
  // CHECK-NEXT: end // always @(edge)
  sv.always edge %clock {
  }

  // CHECK-NEXT: always @* begin
  // CHECK-NEXT: end // always
  sv.always {
  }

  // CHECK-NEXT: always @(posedge clock or negedge cond) begin
  // CHECK-NEXT: end // always @(posedge, negedge)
  sv.always posedge %clock, negedge %cond {
  }

  // CHECK-NEXT: always_ff @(posedge clock)
  // CHECK-NEXT:   $fwrite(32'h80000002, "Yo\n");
  sv.alwaysff(posedge %clock) {
    sv.fwrite %fd, "Yo\n"
  }

  // CHECK-NEXT: always_ff @(posedge clock) begin
  // CHECK-NEXT:   if (cond)
  // CHECK-NEXT:     $fwrite(32'h80000002, "Sync Reset Block\n")
  // CHECK-NEXT:   else
  // CHECK-NEXT:     $fwrite(32'h80000002, "Sync Main Block\n");
  // CHECK-NEXT: end // always_ff @(posedge)
  sv.alwaysff(posedge %clock) {
    sv.fwrite %fd, "Sync Main Block\n"
  } ( syncreset : posedge %cond) {
    sv.fwrite %fd, "Sync Reset Block\n"
  }

  // CHECK-NEXT: always_ff @(posedge clock or negedge cond) begin
  // CHECK-NEXT:   if (!cond)
  // CHECK-NEXT:     $fwrite(32'h80000002, "Async Reset Block\n");
  // CHECK-NEXT:   else
  // CHECK-NEXT:     $fwrite(32'h80000002, "Async Main Block\n");
  // CHECK-NEXT: end // always_ff @(posedge or negedge)
  sv.alwaysff(posedge %clock) {
    sv.fwrite %fd, "Async Main Block\n"
  } ( asyncreset : negedge %cond) {
    sv.fwrite %fd, "Async Reset Block\n"
  }
  // CHECK-NEXT:  (* sv attr *)
  // CHECK-NEXT:  initial begin
  sv.initial {
    // CHECK-NEXT:   if (cond)
    sv.if %cond {
      %c42 = hw.constant 42 : i42
      // CHECK-NEXT: wire42 = 42'h2A;
      sv.bpassign %wire42, %c42 : i42
      %c40 = hw.constant 42 : i40

      %c2_i3 = hw.constant 2 : i3
      // CHECK-NEXT: partSelectReg[3'h2 +: 40] = 40'h2A;
      %a = sv.indexed_part_select_inout %partSelectReg[%c2_i3 : 40] : !hw.inout<i42>, i3
      sv.bpassign %a, %c40 : i40
      // CHECK-NEXT: partSelectReg[3'h2 -: 40] = 40'h2A;
      %b = sv.indexed_part_select_inout %partSelectReg[%c2_i3 decrement: 40] : !hw.inout<i42>, i3
      sv.bpassign %b, %c40 : i40
    } else {
      // CHECK: wire42 = param_y;
      sv.bpassign %wire42, %param_y : i42
    }

    // CHECK-NEXT:   if (cond)
    // CHECK-NOT: begin
    sv.if %cond {
      %c42 = hw.constant 42 : i8
      %add = comb.add %val, %c42 : i8
      %sub_inner = comb.sub %val, %c42 : i8
      %sub = comb.sub %sub_inner, %c42 : i8

      // CHECK-NEXT: $fwrite(32'h80000002, "Inlined! %x %x\n", 8'(val + 8'h2A),
      // CHECK-NEXT:         8'(8'(val - 8'h2A) - 8'h2A));
      sv.fwrite %fd, "Inlined! %x %x\n"(%add, %sub) : i8, i8
    }

    // begin/end required here to avoid else-confusion.

    // CHECK-NEXT:   if (cond) begin
    sv.if %cond {
      // CHECK-NEXT: if (clock)
      sv.if %clock {
        // CHECK-NEXT: $fwrite(32'h80000002, "Inside Block\n");
        sv.fwrite %fd, "Inside Block\n"
      }
      // CHECK-NEXT: end
    } else { // CHECK-NEXT: else
      // CHECK-NOT: begin
      // CHECK-NEXT: $fwrite(32'h80000002, "Else Block\n");
      sv.fwrite %fd, "Else Block\n"
    }

    // CHECK-NEXT:   if (cond) begin
    sv.if %cond {
      // CHECK-NEXT:     $fwrite(32'h80000002, "Hi\n");
      sv.fwrite %fd, "Hi\n"

      // CHECK-NEXT:     $fwrite(32'h80000002, "Bye %x\n", 8'(val + val));
      %tmp = comb.add %val, %val : i8
      sv.fwrite %fd, "Bye %x\n"(%tmp) : i8

      // CHECK-NEXT:     assert(cond);
      sv.assert %cond, immediate
      // CHECK-NEXT:     assert #0 (cond);
      sv.assert %cond, observed
      // CHECK-NEXT:     assert final (cond);
      sv.assert %cond, final
      // CHECK-NEXT:     assert_0: assert(cond);
      sv.assert %cond, immediate label "assert_0"
      // CHECK-NEXT:     assert(cond) else $error("expected %d", val);
      sv.assert %cond, immediate message "expected %d"(%val) : i8

      // CHECK-NEXT:     assume(cond);
      sv.assume %cond, immediate
      // CHECK-NEXT:     assume #0 (cond);
      sv.assume %cond, observed
      // CHECK-NEXT:     assume final (cond);
      sv.assume %cond, final
      // CHECK-NEXT:     assume_0: assume(cond);
      sv.assume %cond, immediate label "assume_0"
      // CHECK-NEXT:     assume(cond) else $error("expected %d", val);
      sv.assume %cond, immediate message "expected %d"(%val) : i8

      // CHECK-NEXT:     cover(cond);
      sv.cover %cond, immediate
      // CHECK-NEXT:     cover #0 (cond);
      sv.cover %cond, observed
      // CHECK-NEXT:     cover final (cond);
      sv.cover %cond, final
      // CHECK-NEXT:     cover_0: cover(cond);
      sv.cover %cond, immediate label "cover_0"

      // Simulator Control Tasks
      // CHECK-NEXT: $stop;
      // CHECK-NEXT: $stop(0);
      sv.stop 1
      sv.stop 0
      // CHECK-NEXT: $finish;
      // CHECK-NEXT: $finish(0);
      sv.finish 1
      sv.finish 0
      // CHECK-NEXT: $exit;
      sv.exit

      // Severity Message Tasks
      // CHECK-NEXT: $fatal;
      // CHECK-NEXT: $fatal(1, "foo");
      // CHECK-NEXT: $fatal(1, "foo", val);
      // CHECK-NEXT: $fatal(0);
      // CHECK-NEXT: $fatal(0, "foo");
      // CHECK-NEXT: $fatal(0, "foo", val);
      sv.fatal 1
      sv.fatal 1, "foo"
      sv.fatal 1, "foo"(%val) : i8
      sv.fatal 0
      sv.fatal 0, "foo"
      sv.fatal 0, "foo"(%val) : i8
      // CHECK-NEXT: $error;
      // CHECK-NEXT: $error("foo");
      // CHECK-NEXT: $error("foo", val);
      sv.error
      sv.error "foo"
      sv.error "foo"(%val) : i8
      // CHECK-NEXT: $warning;
      // CHECK-NEXT: $warning("foo");
      // CHECK-NEXT: $warning("foo", val);
      sv.warning
      sv.warning "foo"
      sv.warning "foo"(%val) : i8
      // CHECK-NEXT: $info;
      // CHECK-NEXT: $info("foo");
      // CHECK-NEXT: $info("foo", val);
      sv.info
      sv.info "foo"
      sv.info "foo"(%val) : i8

      // CHECK-NEXT: Emit some stuff in verilog
      // CHECK-NEXT: Great power and responsibility!
      sv.verbatim "// Emit some stuff in verilog\n// Great power and responsibility!"

      %c42 = hw.constant 42 : i8
      %add = comb.add %val, %c42 : i8
      %c42_2 = hw.constant 42 : i8
      %xor = comb.xor %val, %c42_2 : i8
      sv.verbatim "`define MACRO(a, b) a + b"
      // CHECK-NEXT: `define MACRO
      %text = sv.verbatim.expr "`MACRO({{0}}, {{1}})" (%add, %xor): (i8,i8) -> i8

      // CHECK-NEXT: $fwrite(32'h80000002, "M: %x\n", `MACRO(8'(val + 8'h2A), val ^ 8'h2A));
      sv.fwrite %fd, "M: %x\n"(%text) : i8

    }// CHECK-NEXT:   {{end$}}
  } {sv.attributes = [#sv.attribute<"sv attr">]}
  // CHECK-NEXT:  end // initial

  // CHECK-NEXT: assert property (@(posedge clock) cond);
  sv.assert.concurrent posedge %clock, %cond
  // CHECK-NEXT: assert_1: assert property (@(posedge clock) cond);
  sv.assert.concurrent posedge %clock, %cond label "assert_1"
  // CHECK-NEXT: assert property (@(posedge clock) cond) else $error("expected %d", val);
  sv.assert.concurrent posedge %clock, %cond message "expected %d"(%val) : i8

  // CHECK-NEXT: assume property (@(posedge clock) cond);
  sv.assume.concurrent posedge %clock, %cond
  // CHECK-NEXT: assume_1: assume property (@(posedge clock) cond);
  sv.assume.concurrent posedge %clock, %cond label "assume_1"
  // CHECK-NEXT: assume property (@(posedge clock) cond) else $error("expected %d", $sampled(val));
  %sampledVal = "sv.system.sampled"(%val) : (i8) -> i8
  sv.assume.concurrent posedge %clock, %cond message "expected %d"(%sampledVal) : i8

  // CHECK-NEXT: cover property (@(posedge clock) cond);
  sv.cover.concurrent posedge %clock, %cond
  // CHECK-NEXT: cover_1: cover property (@(posedge clock) cond);
  sv.cover.concurrent posedge %clock, %cond label "cover_1"

  // CHECK-NEXT: initial
  // CHECK-NOT: begin
  sv.initial {
    // CHECK-NEXT: $fatal
    sv.fatal 1
  }

  // CHECK-NEXT: initial begin
  sv.initial {
    sv.verbatim "`define THING 1"
    // CHECK-NEXT: automatic logic _GEN;
    // CHECK:     `define THING
    %thing = sv.verbatim.expr "`THING" : () -> i42
    // CHECK-NEXT: wire42 = `THING;
    sv.bpassign %wire42, %thing : i42

    sv.ifdef.procedural "FOO" {
      // CHECK-NEXT: `ifdef FOO
      %c1 = sv.verbatim.expr "\"THING\"" : () -> i1
      sv.fwrite %fd, "%d" (%c1) : i1
      // CHECK-NEXT: fwrite(32'h80000002, "%d", "THING");
      sv.fwrite %fd, "%d" (%c1) : i1
      // CHECK-NEXT: fwrite(32'h80000002, "%d", "THING");
      %c2 = sv.verbatim.expr "\"VERY_LONG_LINE_VERY_LONG_LINE_VERY_LONG_LINE_VERY_LONG_LINE_VERY_LONG_LINE_VERY_LONG_LINE_VERY_LONG_LINE_VERY_LONG_LINE_VERY_LONG_LINE_VERY_LONG_LINE\"" : () -> i1
      // CHECK-NEXT: _GEN =
      // CHECK-NEXT:   "VERY_LONG_LINE_VERY_LONG_LINE_VERY_LONG_LINE_VERY_LONG_LINE_VERY_LONG_LINE_VERY_LONG_LINE_VERY_LONG_LINE_VERY_LONG_LINE_VERY_LONG_LINE_VERY_LONG_LINE";
      // CHECK-NEXT: fwrite(32'h80000002, "%d", _GEN);
      sv.fwrite %fd, "%d" (%c2) : i1
      // CHECK-NEXT: `endif
    }

    // CHECK-NEXT: wire42 = `THING;
    sv.bpassign %wire42, %thing : i42

    // CHECK-NEXT: casez (val)
    sv.case casez %val : i8
    // CHECK-NEXT: 8'b0000001z: begin
    case b0000001z: {
      // CHECK-NEXT: $fwrite(32'h80000002, "a");
      sv.fwrite %fd, "a"
      // CHECK-NEXT: $fwrite(32'h80000002, "b");
      sv.fwrite %fd, "b"
    } // CHECK-NEXT: end
    // CHECK-NEXT: 8'b000000z1:
    // CHECK-NOT: begin
    case b000000z1: {
      // CHECK-NEXT:  $fwrite(32'h80000002, "y");
      sv.fwrite %fd, "y"
    }  // implicit yield is ok.
    // CHECK-NEXT: 8'b00000x11:
    // CHECK-NOT: begin
    case b00000x11: {
      // CHECK-NEXT:  $fwrite(32'h80000002, "z");
      sv.fwrite %fd, "z"
    }  // implicit yield is ok.
    // CHECK-NEXT: default:
    // CHECK-NOT: begin
    default: {
      // CHECK-NEXT:  $fwrite(32'h80000002, "z");
      sv.fwrite %fd, "z"
    } // CHECK-NEXT: endcase

    // CHECK-NEXT: casex (val)
    sv.case casex %val : i8
    // CHECK-NEXT: 8'b0000001z: begin
    case b0000001z: {
      // CHECK-NEXT: $fwrite(32'h80000002, "a");
      sv.fwrite %fd, "a"
      // CHECK-NEXT: $fwrite(32'h80000002, "b");
      sv.fwrite %fd, "b"
    } // CHECK-NEXT: end
    // CHECK-NEXT: 8'b000000z1:
    // CHECK-NOT: begin
    case b000000z1: {
      // CHECK-NEXT:  $fwrite(32'h80000002, "y");
      sv.fwrite %fd, "y"
    }  // implicit yield is ok.
    // CHECK-NEXT: 8'b00000x11:
    // CHECK-NOT: begin
    case b00000x11: {
      // CHECK-NEXT:  $fwrite(32'h80000002, "z");
      sv.fwrite %fd, "z"
    }  // implicit yield is ok.
    // CHECK-NEXT: default:
    // CHECK-NOT: begin
    default: {
      // CHECK-NEXT:  $fwrite(32'h80000002, "z");
      sv.fwrite %fd, "z"
    } // CHECK-NEXT: endcase

    // CHECK-NEXT: case (val)
    sv.case case %val : i8
    // CHECK-NEXT: 8'b0000001z: begin
    case b0000001z: {
      // CHECK-NEXT: $fwrite(32'h80000002, "a");
      sv.fwrite %fd, "a"
      // CHECK-NEXT: $fwrite(32'h80000002, "b");
      sv.fwrite %fd, "b"
    } // CHECK-NEXT: end
    // CHECK-NEXT: 8'b000000z1:
    // CHECK-NOT: begin
    case b000000z1: {
      // CHECK-NEXT:  $fwrite(32'h80000002, "y");
      sv.fwrite %fd, "y"
    }  // implicit yield is ok.
    // CHECK-NEXT: 8'b00000x11:
    // CHECK-NOT: begin
    case b00000x11: {
      // CHECK-NEXT:  $fwrite(32'h80000002, "z");
      sv.fwrite %fd, "z"
    }  // implicit yield is ok.
    // CHECK-NEXT: default:
    // CHECK-NOT: begin
    default: {
      // CHECK-NEXT:  $fwrite(32'h80000002, "z");
      sv.fwrite %fd, "z"
    } // CHECK-NEXT: endcase



   // CHECK-NEXT: casez (cond)
   sv.case casez %cond : i1
   // CHECK-NEXT: 1'b0:
     case b0: {
       // CHECK-NEXT: $fwrite(32'h80000002, "zero");
       sv.fwrite %fd, "zero"
     }
     // CHECK-NEXT: 1'b1:
     case b1: {
       // CHECK-NEXT: $fwrite(32'h80000002, "one");
       sv.fwrite %fd, "one"
     } // CHECK-NEXT: endcase

    // CHECK-NEXT: priority case (cond)
    sv.case priority %cond : i1
    // CHECK-NEXT: default:
    default: {
      // CHECK-NEXT: $fwrite(32'h80000002, "zero");
      sv.fwrite %fd, "zero"
    } // CHECK-NEXT: endcase

    // CHECK-NEXT: unique casez (cond)
    sv.case casez unique %cond : i1
    // CHECK-NEXT: default:
    default: {
      // CHECK-NEXT: $fwrite(32'h80000002, "zero");
      sv.fwrite %fd, "zero"
    } // CHECK-NEXT: endcase
  }// CHECK-NEXT:   {{end // initial$}}

  sv.ifdef "VERILATOR"  {          // CHECK-NEXT: `ifdef VERILATOR
    sv.verbatim "`define Thing2"   // CHECK-NEXT:   `define Thing2
  } else  {                        // CHECK-NEXT: `else
    sv.verbatim "`define Thing1"   // CHECK-NEXT:   `define Thing1
  }                                // CHECK-NEXT: `endif

  %add = comb.add %val, %val : i8

  // CHECK-NEXT: `define STUFF "wire42 (8'(val + val))"
  sv.verbatim "`define STUFF \"{{0}} ({{1}})\"" (%wire42, %add) : !hw.inout<i42>, i8

  // CHECK-NEXT: `ifdef FOO
  sv.ifdef "FOO" {
    %c1 = sv.verbatim.expr "\"THING\"" : () -> i1

    // CHECK-NEXT: initial begin
    sv.initial {
      // CHECK-NEXT: fwrite(32'h80000002, "%d", "THING");
      sv.fwrite %fd, "%d" (%c1) : i1
      // CHECK-NEXT: fwrite(32'h80000002, "%d", "THING");
      sv.fwrite %fd, "%d" (%c1) : i1

    // CHECK-NEXT: end // initial
    }

  // CHECK-NEXT: `endif
  }
}

// CHECK-LABEL: module Aliasing(
// CHECK-NEXT:    inout [41:0] a, //
// CHECK-NEXT:                 b, //
// CHECK-NEXT:                 c //
// CHECK-NEXT:  );
hw.module @Aliasing(%a : !hw.inout<i42>, %b : !hw.inout<i42>,
                      %c : !hw.inout<i42>) {

  // CHECK: alias a = b;
  sv.alias %a, %b     : !hw.inout<i42>, !hw.inout<i42>
  // CHECK: alias a = b = c;
  sv.alias %a, %b, %c : !hw.inout<i42>, !hw.inout<i42>, !hw.inout<i42>
}

hw.module @reg_0(%in4: i4, %in8: i8) -> (a: i8, b: i8) {
  // CHECK-LABEL: module reg_0(
  // CHECK-NEXT:   input  [3:0] in4, //
  // CHECK-NEXT:   input  [7:0] in8, //
  // CHECK-NEXT:   output [7:0] a, //
  // CHECK-NEXT:                b //
  // CHECK-NEXT:  );

  // CHECK-EMPTY:
  // CHECK-NEXT: (* dont_merge *)
  // CHECK-NEXT: reg [7:0]       myReg;
  %myReg = sv.reg {sv.attributes = [#sv.attribute<"dont_merge">]} : !hw.inout<i8>

  // CHECK-NEXT: (* dont_merge, dont_retime = true *)
  // CHECK-NEXT: /* comment_merge, comment_retime = true */
  // CHECK-NEXT: (* another_attr *)
  // CHECK-NEXT: reg [41:0][7:0] myRegArray1;
  %myRegArray1 = sv.reg {sv.attributes = [
    #sv.attribute<"dont_merge">,
    #sv.attribute<"dont_retime" = "true">,
    #sv.attribute<"comment_merge", emitAsComment>,
    #sv.attribute<"comment_retime" = "true", emitAsComment>,
    #sv.attribute<"another_attr">
  ]} : !hw.inout<array<42 x i8>>

  // CHECK:      /* assign_attr */
  // CHECK-NEXT: assign myReg = in8;
  sv.assign %myReg, %in8 {sv.attributes = [#sv.attribute<"assign_attr", emitAsComment>]} : i8

  %subscript1 = sv.array_index_inout %myRegArray1[%in4] : !hw.inout<array<42 x i8>>, i4
  sv.assign %subscript1, %in8 : i8   // CHECK-NEXT: assign myRegArray1[in4] = in8;

  %regout = sv.read_inout %myReg : !hw.inout<i8>

  %subscript2 = sv.array_index_inout %myRegArray1[%in4] : !hw.inout<array<42 x i8>>, i4
  %memout = sv.read_inout %subscript2 : !hw.inout<i8>

  // CHECK-NEXT: assign a = myReg;
  // CHECK-NEXT: assign b = myRegArray1[in4];
  hw.output %regout, %memout : i8, i8
}

hw.module @reg_1(%in4: i4, %in8: i8) -> (a : i3, b : i5) {
  // CHECK-LABEL: module reg_1(

  // CHECK: reg [17:0] myReg2
  %myReg2 = sv.reg : !hw.inout<i18>

  // CHECK:      assign myReg2[4'h7 +: 8] = in8;
  // CHECK-NEXT: assign myReg2[4'h7 -: 8] = in8;

  %c2_i3 = hw.constant 7 : i4
  %a1 = sv.indexed_part_select_inout %myReg2[%c2_i3 : 8] : !hw.inout<i18>, i4
  sv.assign %a1, %in8 : i8
  %b1 = sv.indexed_part_select_inout %myReg2[%c2_i3 decrement: 8] : !hw.inout<i18>, i4
  sv.assign %b1, %in8 : i8
  %c3_i3 = hw.constant 3 : i4
  %r1 = sv.read_inout %myReg2 : !hw.inout<i18>
  %c = sv.indexed_part_select %r1[%c3_i3 : 3] : i18,i4
  %d = sv.indexed_part_select %r1[%in4 decrement:5] :i18, i4
  // CHECK-NEXT: assign a = myReg2[4'h3 +: 3];
  // CHECK-NEXT: assign b = myReg2[in4 -: 5];
  hw.output %c, %d : i3,i5
}

// CHECK-LABEL: module struct_field_inout1(
// CHECK-NEXT:   inout struct packed {logic b; } a
// CHECK-NEXT:  );
hw.module @struct_field_inout1(%a : !hw.inout<struct<b: i1>>) {
  // CHECK: assign a.b = 1'h1;
  %true = hw.constant true
  %0 = sv.struct_field_inout %a["b"] : !hw.inout<struct<b: i1>>
  sv.assign %0, %true : i1
}

// CHECK-LABEL: module struct_field_inout2(
// CHECK-NEXT:    inout struct packed {struct packed {logic c; } b; } a
// CHECK-NEXT:  );
hw.module @struct_field_inout2(%a: !hw.inout<struct<b: !hw.struct<c: i1>>>) {
  // CHECK: assign a.b.c = 1'h1;
  %true = hw.constant true
  %0 = sv.struct_field_inout %a["b"] : !hw.inout<struct<b: !hw.struct<c: i1>>>
  %1 = sv.struct_field_inout %0["c"] : !hw.inout<struct<c: i1>>
  sv.assign %1, %true : i1
}

// CHECK-LABEL: module PartSelectInoutInline(
hw.module @PartSelectInoutInline(%v:i40) {
  %r = sv.reg : !hw.inout<i42>
  %c2_i3 = hw.constant 2 : i3
  %a = sv.indexed_part_select_inout %r[%c2_i3 : 40] : !hw.inout<i42>, i3
  // CHECK: initial
  // CHECK-NEXT:   r[3'h2 +: 40] = v;
  sv.initial {
    sv.bpassign %a, %v : i40
  }
}

// CHECK-LABEL: module AggregateConstantXZ(
hw.module @AggregateConstantXZ() -> (res1: !hw.struct<foo: i2, bar: !hw.array<3xi4>>,
                                     res2: !hw.struct<foo: i2, bar: !hw.array<3xi4>>) {
  %0 = sv.constantX : !hw.struct<foo: i2, bar: !hw.array<3xi4>>
  %1 = sv.constantZ : !hw.struct<foo: i2, bar: !hw.array<3xi4>>
  // CHECK: assign res1 = 14'bx
  // CHECK: assign res2 = 14'bz
  hw.output %0, %1 : !hw.struct<foo: i2, bar: !hw.array<3xi4>>, !hw.struct<foo: i2, bar: !hw.array<3xi4>>
}

// CHECK-LABEL: module AggregateVerbatim(
hw.module @AggregateVerbatim() -> (res1: !hw.struct<a: i1>, res2: !hw.array<1xi1>, res3: !hw.array<1xi1>) {
  %a = sv.verbatim.expr "STRUCT_A_" : () -> !hw.struct<a: i1>
  %b = sv.verbatim.expr "ARRAY_" : () -> !hw.array<1xi1>
  %c = sv.verbatim.expr "MACRO({{0}}, {{1}})" (%a, %b) : (!hw.struct<a: i1>, !hw.array<1xi1>) -> !hw.array<1xi1>
  hw.output %a, %b, %c: !hw.struct<a: i1>, !hw.array<1xi1>, !hw.array<1xi1>
  // CHECK: assign res1 = STRUCT_A_;
  // CHECK: assign res2 = ARRAY_;
  // CHECK: assign res3 = MACRO(STRUCT_A_, ARRAY_);
}

// CHECK-LABEL: issue508
// https://github.com/llvm/circt/issues/508
hw.module @issue508(%in1: i1, %in2: i1) {
  // CHECK: wire _GEN = in1 | in2;
  %clock = comb.or %in1, %in2 : i1

  // CHECK-NEXT: always @(posedge _GEN) begin
  // CHECK-NEXT: end
  sv.always posedge %clock {
  }
}

// CHECK-LABEL: exprInlineTestIssue439
// https://github.com/llvm/circt/issues/439
hw.module @exprInlineTestIssue439(%clk: i1) {
  %fd = hw.constant 0x80000002 : i32

  // CHECK: always @(posedge clk) begin
  sv.always posedge %clk {
    %c = hw.constant 0 : i32

    // CHECK: automatic logic [31:0] _GEN = 32'h0;
    %e = comb.extract %c from 0 : (i32) -> i16
    %f = comb.add %e, %e : i16
    sv.fwrite %fd, "%d"(%f) : i16
    // CHECK: $fwrite(32'h80000002, "%d", 16'(_GEN[15:0] + _GEN[15:0]));
    // CHECK: end // always @(posedge)
  }
}

// https://github.com/llvm/circt/issues/595
// CHECK-LABEL: module issue595
hw.module @issue595(%arr: !hw.array<128xi1>) {
  // CHECK: wire [31:0] [[TEMP1:.+]];
  %c0_i32 = hw.constant 0 : i32
  %c0_i7 = hw.constant 0 : i7
  %c0_i6 = hw.constant 0 : i6
  %0 = comb.icmp eq %3, %c0_i32 : i32

  sv.initial {
    // CHECK: assert([[TEMP1]] == 32'h0);
    sv.assert %0, immediate
  }

  // CHECK: wire [63:0] [[TEMP2:.+]] = arr[7'h0 +: 64];
  // CHECK: assign [[TEMP1]] = [[TEMP2:.+]][6'h0 +: 32];
  %1 = hw.array_slice %arr[%c0_i7] : (!hw.array<128xi1>) -> !hw.array<64xi1>
  %2 = hw.array_slice %1[%c0_i6] : (!hw.array<64xi1>) -> !hw.array<32xi1>
  %3 = hw.bitcast %2 : (!hw.array<32xi1>) -> i32
  hw.output
}


// CHECK-LABEL: module issue595_variant1
hw.module @issue595_variant1(%arr: !hw.array<128xi1>) {
  // CHECK: wire [31:0] [[TEMP1:.+]];
  %c0_i32 = hw.constant 0 : i32
  %c0_i7 = hw.constant 0 : i7
  %c0_i6 = hw.constant 0 : i6
  %0 = comb.icmp ne %3, %c0_i32 : i32

  sv.initial {
    // CHECK: assert(|[[TEMP1]]);
    sv.assert %0, immediate
  }

  // CHECK: wire [63:0] [[TEMP2:.+]] = arr[7'h0 +: 64];
  // CHECK: assign [[TEMP1]] = [[TEMP2]][6'h0 +: 32];
  %1 = hw.array_slice %arr[%c0_i7] : (!hw.array<128xi1>) -> !hw.array<64xi1>
  %2 = hw.array_slice %1[%c0_i6] : (!hw.array<64xi1>) -> !hw.array<32xi1>
  %3 = hw.bitcast %2 : (!hw.array<32xi1>) -> i32
  hw.output
}

// CHECK-LABEL: module issue595_variant2_checkRedunctionAnd
hw.module @issue595_variant2_checkRedunctionAnd(%arr: !hw.array<128xi1>) {
  // CHECK: wire [31:0] [[TEMP1:.+]];
  %c0_i32 = hw.constant -1 : i32
  %c0_i7 = hw.constant 0 : i7
  %c0_i6 = hw.constant 0 : i6
  %0 = comb.icmp eq %3, %c0_i32 : i32

  sv.initial {
    // CHECK: assert(&[[TEMP1]]);
    sv.assert %0, immediate
  }

  // CHECK: wire [63:0] [[TEMP2:.+]] = arr[7'h0 +: 64];
  // CHECK: assign _GEN = [[TEMP2]][6'h0 +: 32];
  %1 = hw.array_slice %arr[%c0_i7] : (!hw.array<128xi1>) -> !hw.array<64xi1>
  %2 = hw.array_slice %1[%c0_i6] : (!hw.array<64xi1>) -> !hw.array<32xi1>
  %3 = hw.bitcast %2 : (!hw.array<32xi1>) -> i32
  hw.output
}

// CHECK-LABEL: module slice_inline_ports
hw.module @slice_inline_ports(%arr: !hw.array<128xi1>, %x: i3, %y: i7)
 -> (o1: !hw.array<2xi3>, o2: !hw.array<64xi1>, o3: !hw.array<64xi1>) {

  // array_create cannot be inlined into the slice.
  %c1_i2 = hw.constant 1 : i2
  %0 = hw.array_create %x, %x, %x, %x : i3
  // CHECK: wire [3:0][2:0] _GEN =
  %1 = hw.array_slice %0[%c1_i2] : (!hw.array<4xi3>) -> !hw.array<2xi3>
  // CHECK: assign o1 = _GEN[2'h1 +: 2];

  %c1_i7 = hw.constant 1 : i7

  /// This can be inlined.
  // CHECK: assign o2 = arr[7'h1 +: 64];
  %2 = hw.array_slice %arr[%c1_i7] : (!hw.array<128xi1>) -> !hw.array<64xi1>

  // CHECK: assign o3 = arr[7'(y + 7'h1) +: 64];
  %sum = comb.add %y, %c1_i7 : i7
  %3 = hw.array_slice %arr[%sum] : (!hw.array<128xi1>) -> !hw.array<64xi1>

  hw.output %1, %2, %3: !hw.array<2xi3>, !hw.array<64xi1>, !hw.array<64xi1>
}



// CHECK-LABEL: if_multi_line_expr1
hw.module @if_multi_line_expr1(%clock: i1, %reset: i1, %really_long_port: i11) {
  %tmp6 = sv.reg  : !hw.inout<i25>

  // CHECK: if (reset)
  // CHECK-NEXT:   tmp6 <= 25'h0;
  // CHECK-NEXT: else
  // CHECK-NEXT:   tmp6 <= {{..}}14{really_long_port[10]}}, really_long_port} & 25'h3039;
  // CHECK-NEXT: end
  sv.alwaysff(posedge %clock) {
    %sign = comb.extract %really_long_port from 10 : (i11) -> i1
    %signs = comb.replicate %sign : (i1) -> i14
    %0 = comb.concat %signs, %really_long_port : i14, i11
    %c12345_i25 = hw.constant 12345 : i25
    %1 = comb.and %0, %c12345_i25 : i25
    sv.passign %tmp6, %1 : i25
  }(syncreset : posedge %reset)  {
    %c0_i25 = hw.constant 0 : i25
    sv.passign %tmp6, %c0_i25 : i25
  }
  hw.output
}

// CHECK-LABEL: if_multi_line_expr2
hw.module @if_multi_line_expr2(%clock: i1, %reset: i1, %really_long_port: i11) {
  %tmp6 = sv.reg  : !hw.inout<i25>

  %c12345_i25 = hw.constant 12345 : i25
  %sign = comb.extract %really_long_port from 10 : (i11) -> i1
  %signs = comb.replicate %sign : (i1) -> i14
  %0 = comb.concat %signs, %really_long_port : i14, i11
  %1 = comb.and %0, %c12345_i25 : i25

  // CHECK:      if (reset)
  // CHECK-NEXT:   tmp6 <= 25'h0;
  // CHECK-NEXT: else
  // CHECK-NEXT:   tmp6 <= {{..}}14{really_long_port[10]}}, really_long_port} & 25'h3039;
  sv.alwaysff(posedge %clock)  {
    sv.passign %tmp6, %1 : i25
  }(syncreset : posedge %reset)  {
    %c0_i25 = hw.constant 0 : i25
    sv.passign %tmp6, %c0_i25 : i25
  }
  hw.output
}

// https://github.com/llvm/circt/issues/720
// CHECK-LABEL: module issue720(
hw.module @issue720(%clock: i1, %arg1: i1, %arg2: i1, %arg3: i1) {

  // CHECK: always @(posedge clock) begin
  sv.always posedge %clock  {
    // CHECK:   automatic logic _GEN = arg1 & arg2;

    // CHECK:   if (arg1)
    // CHECK:     $fatal;
    sv.if %arg1  {
      sv.fatal 1
    }

    // CHECK:   if (_GEN)
    // CHECK:     $fatal;

    //this forces a common subexpression to be output out-of-line
    %610 = comb.and %arg1, %arg2 : i1
    %611 = comb.and %arg3, %610 : i1
    sv.if %610  {
      sv.fatal 1
    }

    // CHECK:   if (arg3 & _GEN)
    // CHECK:     $fatal;
    sv.if %611  {
      sv.fatal 1
    }
  } // CHECK: end // always @(posedge)

  hw.output
}

// CHECK-LABEL: module issue720ifdef(
hw.module @issue720ifdef(%clock: i1, %arg1: i1, %arg2: i1, %arg3: i1) {
  // CHECK: always @(posedge clock) begin
  sv.always posedge %clock  {
    // The variable for the ifdef block needs to be emitted at the top of the
    // always block since the ifdef is transparent to verilog.

    // CHECK:    automatic logic _GEN;
    // CHECK:    if (arg1)
    // CHECK:      $fatal;
    sv.if %arg1  {
      sv.fatal 1
    }

    // CHECK:    `ifdef FUN_AND_GAMES
     sv.ifdef.procedural "FUN_AND_GAMES" {
      // This forces a common subexpression to be output out-of-line
      // CHECK:      _GEN = arg1 & arg2;
      // CHECK:      if (_GEN)
      // CHECK:        $fatal;
      %610 = comb.and %arg1, %arg2 : i1
      sv.if %610  {
        sv.fatal 1
      }
      // CHECK:      if (arg3 & _GEN)
      // CHECK:        $fatal;
      %611 = comb.and %arg3, %610 : i1
     sv.if %611  {
        sv.fatal 1
      }
      // CHECK:    `endif
      // CHECK:  end // always @(posedge)
    }
  }
  hw.output
}

// https://github.com/llvm/circt/issues/728

// CHECK-LABEL: module issue728(
hw.module @issue728(%clock: i1, %asdfasdfasdfasdfafa: i1, %gasfdasafwjhijjafija: i1) {
  %fd = hw.constant 0x80000002 : i32

  // CHECK:       always @(posedge clock) begin
  // CHECK-NEXT:    $fwrite(32'h80000002, "force output");
  // CHECK-NEXT:    if (asdfasdfasdfasdfafa & gasfdasafwjhijjafija & asdfasdfasdfasdfafa
  // CHECK-NEXT:        & gasfdasafwjhijjafija & asdfasdfasdfasdfafa & gasfdasafwjhijjafija)
  // CHECK-NEXT:      $fwrite(32'h80000002, "this cond is split");
  // CHECK-NEXT:  end // always @(posedge)
  sv.always posedge %clock  {
     sv.fwrite %fd, "force output"
     %cond = comb.and %asdfasdfasdfasdfafa, %gasfdasafwjhijjafija, %asdfasdfasdfasdfafa, %gasfdasafwjhijjafija, %asdfasdfasdfasdfafa, %gasfdasafwjhijjafija : i1
     sv.if %cond  {
       sv.fwrite %fd, "this cond is split"
     }
  }
  hw.output
}

// CHECK-LABEL: module issue728ifdef(
hw.module @issue728ifdef(%clock: i1, %asdfasdfasdfasdfafa: i1, %gasfdasafwjhijjafija: i1) {
  %fd = hw.constant 0x80000002 : i32

  // CHECK:      always @(posedge clock) begin
  // CHECK-NEXT:    $fwrite(32'h80000002, "force output");
  // CHECK-NEXT:    `ifdef FUN_AND_GAMES
  // CHECK-NEXT:    if (asdfasdfasdfasdfafa & gasfdasafwjhijjafija & asdfasdfasdfasdfafa
  // CHECK-NEXT:         & gasfdasafwjhijjafija & asdfasdfasdfasdfafa & gasfdasafwjhijjafija)
  // CHECK-NEXT:        $fwrite(32'h80000002, "this cond is split");
  // CHECK-NEXT:    `endif
  // CHECK-NEXT: end // always @(posedge)
  sv.always posedge %clock  {
     sv.fwrite %fd, "force output"
     sv.ifdef.procedural "FUN_AND_GAMES" {
       %cond = comb.and %asdfasdfasdfasdfafa, %gasfdasafwjhijjafija, %asdfasdfasdfasdfafa, %gasfdasafwjhijjafija, %asdfasdfasdfasdfafa, %gasfdasafwjhijjafija : i1
       sv.if %cond  {
         sv.fwrite %fd, "this cond is split"
       }
     }
  }
}

// CHECK-LABEL: module alwayscombTest(
hw.module @alwayscombTest(%a: i1) -> (x: i1) {
  // CHECK: reg combWire;
  %combWire = sv.reg : !hw.inout<i1>
  // CHECK: always_comb
  sv.alwayscomb {
    // CHECK-NEXT: combWire <= a
    sv.passign %combWire, %a : i1
  }

  // CHECK: assign x = combWire;
  %out = sv.read_inout %combWire : !hw.inout<i1>
  hw.output %out : i1
}


// https://github.com/llvm/circt/issues/838
// CHECK-LABEL: module inlineProceduralWiresWithLongNames(
hw.module @inlineProceduralWiresWithLongNames(%clock: i1, %in: i1) {
  %aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa = sv.wire  : !hw.inout<i1>
  %0 = sv.read_inout %aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa : !hw.inout<i1>
  %bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb = sv.wire  : !hw.inout<i1>
  %1 = sv.read_inout %bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb : !hw.inout<i1>
  %r = sv.reg  : !hw.inout<uarray<1xi1>>
  %s = sv.reg  : !hw.inout<uarray<1xi1>>
  %2 = sv.array_index_inout %r[%0] : !hw.inout<uarray<1xi1>>, i1
  %3 = sv.array_index_inout %s[%1] : !hw.inout<uarray<1xi1>>, i1
  // CHECK: always_ff
  sv.alwaysff(posedge %clock)  {
    // CHECK-NEXT: r[aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa] <= in;
    sv.passign %2, %in : i1
    // CHECK-NEXT: s[bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb] <= in;
    sv.passign %3, %in : i1
  }
}

// https://github.com/llvm/circt/issues/859
// CHECK-LABEL: module oooReg(
hw.module @oooReg(%in: i1) -> (result: i1) {
  // CHECK: wire abc = in;
  %0 = sv.read_inout %abc : !hw.inout<i1>

  sv.assign %abc, %in : i1
  %abc = sv.wire  : !hw.inout<i1>

  // CHECK: assign result = abc;
  hw.output %0 : i1
}

// https://github.com/llvm/circt/issues/865
// CHECK-LABEL: module ifdef_beginend(
hw.module @ifdef_beginend(%clock: i1, %cond: i1, %val: i8) {
  // CHECK: always @(posedge clock) begin
  sv.always posedge %clock  {
    // CHECK-NEXT: `ifndef SYNTHESIS
    sv.ifdef.procedural "SYNTHESIS"  {
    } // CHECK-NEXT: `endif
  } // CHECK-NEXT: end
} // CHECK-NEXT: endmodule

// https://github.com/llvm/circt/issues/884
// CHECK-LABEL: module ConstResetValueMustBeInlined(
hw.module @ConstResetValueMustBeInlined(%clock: i1, %reset: i1, %d: i42) -> (q: i42) {
  %c0_i42 = hw.constant 0 : i42
  %tmp = sv.reg : !hw.inout<i42>
  // CHECK: always_ff @(posedge clock or posedge reset) begin
  // CHECK-NEXT:   if (reset)
  // CHECK-NEXT:     tmp <= 42'h0;
  sv.alwaysff(posedge %clock) {
    sv.passign %tmp, %d : i42
  } (asyncreset : posedge %reset)  {
    sv.passign %tmp, %c0_i42 : i42
  }
  %1 = sv.read_inout %tmp : !hw.inout<i42>
  hw.output %1 : i42
}

// CHECK-LABEL: module OutOfLineConstantsInAlwaysSensitivity
hw.module @OutOfLineConstantsInAlwaysSensitivity() {
  // CHECK-NEXT: wire _GEN = 1'h0;
  // CHECK-NEXT: always_ff @(posedge _GEN)
  %clock = hw.constant 0 : i1
  sv.alwaysff(posedge %clock) {}
}

// CHECK-LABEL: module TooLongConstExpr
hw.module @TooLongConstExpr() {
  %myreg = sv.reg : !hw.inout<i4200>
  // CHECK: always @*
  sv.always {
    // CHECK-NEXT: myreg <=
    // CHECK-NEXT:   4200'(4200'h2323CB3A9903AD1D87D91023532E89D313E12BFCFCA2492A8561CADD94652CC4
    // CHECK-NEXT:         + 4200'h2323CB3A9903AD1D87D91023532E89D313E12BFCFCA2492A8561CADD94652CC4);
    %0 = hw.constant 15894191981981165163143546843135416146464164161464654561818646486465164684484 : i4200
    %1 = comb.add %0, %0 : i4200
    sv.passign %myreg, %1 : i4200
  }
}

// Constants defined before use should be emitted in-place.
// CHECK-LABEL: module ConstantDefBeforeUse
hw.module @ConstantDefBeforeUse() {
  %myreg = sv.reg : !hw.inout<i32>
  // CHECK: always @*
  // CHECK-NEXT:   myreg <= 32'h2A;
  %0 = hw.constant 42 : i32
  sv.always {
    sv.passign %myreg, %0 : i32
  }
}

// Constants defined after use in non-procedural regions should be moved to the
// top of the block.
// CHECK-LABEL: module ConstantDefAfterUse
hw.module @ConstantDefAfterUse() {
  %myreg = sv.reg : !hw.inout<i32>
  // CHECK: always @*
  // CHECK-NEXT:   myreg <= 32'h2A;
  sv.always {
    sv.passign %myreg, %0 : i32
  }
  %0 = hw.constant 42 : i32
}

// Constants defined in a procedural block with users in a different block
// should be emitted at the top of their defining block.
// CHECK-LABEL: module ConstantEmissionAtTopOfBlock
hw.module @ConstantEmissionAtTopOfBlock() {
  %myreg = sv.reg : !hw.inout<i32>
  // CHECK:      always @* begin
  // CHECK-NEXT:   if (1'h1)
  // CHECK-NEXT:     myreg <= 32'h2A;
  sv.always {
    %0 = hw.constant 42 : i32
    %1 = hw.constant 1 : i1
    sv.if %1 {
      sv.passign %myreg, %0 : i32
    }
  }
}

// See https://github.com/llvm/circt/issues/1356
// CHECK-LABEL: module RegisterOfStructOrArrayOfStruct
hw.module @RegisterOfStructOrArrayOfStruct() {
  // CHECK-NOT: reg
  // CHECK: struct packed {logic a; logic b; }           reg1
  %reg1 = sv.reg : !hw.inout<struct<a: i1, b: i1>>

  // CHECK-NOT: reg
  // CHECK: struct packed {logic a; logic b; }[7:0]      reg2
  %reg2 = sv.reg : !hw.inout<array<8xstruct<a: i1, b: i1>>>

  // CHECK-NOT: reg
  // CHECK: struct packed {logic a; logic b; }[3:0][7:0] reg3
  %reg3 = sv.reg : !hw.inout<array<4xarray<8xstruct<a: i1, b: i1>>>>
}


// CHECK-LABEL: module MultiUseReadInOut(
// Issue #1564
hw.module @MultiUseReadInOut(%auto_in_ar_bits_id : i2) -> (aa: i3, bb: i3){
  %a = sv.reg  : !hw.inout<i3>
  %b = sv.reg  : !hw.inout<i3>
  %c = sv.reg  : !hw.inout<i3>
  %d = sv.reg  : !hw.inout<i3>
  %123 = sv.read_inout %b : !hw.inout<i3>
  %124 = sv.read_inout %a : !hw.inout<i3>
  %125 = sv.read_inout %c : !hw.inout<i3>
  %126 = sv.read_inout %d : !hw.inout<i3>

  // We should directly use a/b/c/d here instead of emitting temporary wires.

  // CHECK: wire [3:0][2:0] [[WIRE:.+]] = {{.}}{a}, {b}, {c}, {d}};
  // CHECK-NEXT: assign aa = [[WIRE]][auto_in_ar_bits_id];
  %127 = hw.array_create %124, %123, %125, %126 : i3
  %128 = hw.array_get %127[%auto_in_ar_bits_id] : !hw.array<4xi3>, i2

  // CHECK: assign bb = 3'(b + a);
  %xx = comb.add %123, %124 : i3
  hw.output %128, %xx : i3, i3
}

// CHECK-LABEL: module DontDuplicateSideEffectingVerbatim(
hw.module @DontDuplicateSideEffectingVerbatim() {
  %a = sv.reg : !hw.inout<i42>
  %b = sv.reg sym @regSym : !hw.inout<i42>

  sv.initial {
    // CHECK: automatic logic [41:0] _SIDEEFFECT = SIDEEFFECT;
    // CHECK-NEXT: automatic logic [41:0] _GEN = b;
    %tmp = sv.verbatim.expr.se "SIDEEFFECT" : () -> i42
    %verb_tmp = sv.verbatim.expr.se "{{0}}" : () -> i42 {symbols = [#hw.innerNameRef<@DontDuplicateSideEffectingVerbatim::@regSym>]}
    // CHECK: a = _SIDEEFFECT;
    sv.bpassign %a, %tmp : i42
    // CHECK: a = _SIDEEFFECT;
    sv.bpassign %a, %tmp : i42

    // CHECK: a = _GEN;
    sv.bpassign %a, %verb_tmp : i42
    // CHECK: a = _GEN;
    sv.bpassign %a, %verb_tmp : i42
    %tmp2 = sv.verbatim.expr "NO_EFFECT_" : () -> i42
    // CHECK: a = NO_EFFECT_;
    sv.bpassign %a, %tmp2 : i42
    // CHECK: a = NO_EFFECT_;
    sv.bpassign %a, %tmp2 : i42
  }
}

hw.generator.schema @verbatim_schema, "Simple", ["ports", "write_latency", "read_latency"]
hw.module.extern @verbatim_inout_2 () -> ()
// CHECK-LABEL: module verbatim_M1(
hw.module @verbatim_M1(%clock : i1, %cond : i1, %val : i8) {
  %c42 = hw.constant 42 : i8
  %reg1 = sv.reg sym @verbatim_reg1: !hw.inout<i8>
  %reg2 = sv.reg sym @verbatim_reg2: !hw.inout<i8>
  // CHECK:      (* dont_merge *)
  // CHECK-NEXT: wire [22:0] wire25
  %wire25 = sv.wire sym @verbatim_wireSym1 {sv.attributes = [#sv.attribute<"dont_merge">]} : !hw.inout<i23>
  %add = comb.add %val, %c42 : i8
  %c42_2 = hw.constant 42 : i8
  %xor = comb.xor %val, %c42_2 : i8
  hw.instance "aa1" sym @verbatim_b1 @verbatim_inout_2() ->()
  // CHECK: MACRO(8'(val + 8'h2A), val ^ 8'h2A reg=reg1, verbatim_M2, verbatim_inout_2, aa1,reg2 = reg2 )
  sv.verbatim  "MACRO({{0}}, {{1}} reg={{2}}, {{3}}, {{4}}, {{5}},reg2 = {{6}} )"
          (%add, %xor)  : i8,i8
          {symbols = [#hw.innerNameRef<@verbatim_M1::@verbatim_reg1>, @verbatim_M2,
          @verbatim_inout_2, #hw.innerNameRef<@verbatim_M1::@verbatim_b1>, #hw.innerNameRef<@verbatim_M1::@verbatim_reg2>]}
  // CHECK: Wire : wire25
  sv.verbatim " Wire : {{0}}" {symbols = [#hw.innerNameRef<@verbatim_M1::@verbatim_wireSym1>]}
}

// CHECK-LABEL: module verbatim_M2(
hw.module @verbatim_M2(%clock : i1, %cond : i1, %val : i8) {
  %c42 = hw.constant 42 : i8
  %add = comb.add %val, %c42 : i8
  %c42_2 = hw.constant 42 : i8
  %xor = comb.xor %val, %c42_2 : i8
  // CHECK: MACRO(8'(val + 8'h2A), val ^ 8'h2A, verbatim_M1 -- verbatim_M2)
  sv.verbatim  "MACRO({{0}}, {{1}}, {{2}} -- {{3}})"
                (%add, %xor)  : i8,i8
                {symbols = [@verbatim_M1, @verbatim_M2, #hw.innerNameRef<@verbatim_M1::@verbatim_b1>]}
}

// CHECK-LABEL: module InlineAutomaticLogicInit(
// Issue #1567: https://github.com/llvm/circt/issues/1567
hw.module @InlineAutomaticLogicInit(%a : i42, %b: i42, %really_really_long_port: i11) {
  %regValue = sv.reg : !hw.inout<i42>
  // CHECK: initial begin
  sv.initial {
    // CHECK-DAG: automatic logic [63:0] [[_THING:.+]] = `THING;
    // CHECK-DAG: automatic logic [41:0] [[GEN_0:.+]] = 42'(a + a);
    // CHECK-DAG: automatic logic [41:0] [[GEN_1:.+]] = 42'([[GEN_0]] + b);
    // CHECK-DAG: automatic logic [41:0] [[GEN_2:.+]];
    %thing = sv.verbatim.expr "`THING" : () -> i64

    // CHECK: regValue = _THING[44:3];
    %v = comb.extract %thing from 3 : (i64) -> i42
    sv.bpassign %regValue, %v : i42

    // tmp is multi-use, so it needs an 'automatic logic'.  This can be emitted
    // inline because it just references ports.
    %tmp = comb.add %a, %a : i42
    sv.bpassign %regValue, %tmp : i42
    // CHECK: regValue = [[GEN_0]];

    // tmp2 is as well.  This can be emitted inline because it just references
    // a port and an already-emitted-inline variable 'a'.
    %tmp2 = comb.add %tmp, %b : i42
    sv.bpassign %regValue, %tmp2 : i42
    // CHECK: regValue = [[GEN_1]];

    %tmp3 = comb.add %tmp2, %b : i42
    sv.bpassign %regValue, %tmp3 : i42
    // CHECK: regValue = 42'([[GEN_1]] + b);

    // CHECK: `ifdef FOO
    sv.ifdef.procedural "FOO" {
      // CHECK: [[GEN_2]] = 42'(a + a);
      // tmp is multi-use so it needs a temporary, but cannot be emitted inline
      // because it is in an ifdef.
      %tmp4 = comb.add %a, %a : i42
      sv.bpassign %regValue, %tmp4 : i42
      // CHECK: regValue = [[GEN_2]];

      %tmp5 = comb.add %tmp4, %b : i42
      sv.bpassign %regValue, %tmp5 : i42
      // CHECK: regValue = 42'([[GEN_2]] + b);
    }
  }

  // Check that inline initializer things can have too-long-line-length
  // temporaries and that they are generated correctly.

  // CHECK: initial begin
  sv.initial {
    // CHECK: automatic logic [41:0] [[THING:.+]];
    // CHECK: automatic logic [41:0] [[THING3:.+]];
    // CHECK: automatic logic [41:0] [[MANYTHING:.+]];
    // CHECK: [[THING]] = `THING;
    // CHECK: [[THING3]] = 42'([[THING]] + {{..}}31{really_really_long_port[10]}}
    // CHECK-SAME: really_really_long_port})
    // CHECK: [[MANYTHING]] =
    // CHECK-NEXT: [[THING]] | [[THING]] |

    // Check the indentation level of temporaries.  Issue #1625
    %thing = sv.verbatim.expr.se "`THING" : () -> i42

    %sign = comb.extract %really_really_long_port from 10 : (i11) -> i1
    %signs = comb.replicate %sign : (i1) -> i31
    %thing2 = comb.concat %signs, %really_really_long_port : i31, i11
    %thing3 = comb.add %thing, %thing2 : i42  // multiuse.

    // multiuse, refers to other 'automatic logic' thing so must be emitted in
    // the proper order.
    %manyThing = comb.or %thing, %thing, %thing, %thing, %thing, %thing,
                         %thing, %thing, %thing, %thing, %thing, %thing,
                         %thing, %thing, %thing, %thing, %thing, %thing,
                         %thing, %thing, %thing, %thing, %thing, %thing : i42

    // CHECK: regValue = [[THING]];
    sv.bpassign %regValue, %thing : i42
    // CHECK: regValue = [[THING3]];
    sv.bpassign %regValue, %thing3 : i42
    // CHECK: regValue = [[THING3]];
    sv.bpassign %regValue, %thing3 : i42
    // CHECK: regValue = [[MANYTHING]];
    sv.bpassign %regValue, %manyThing : i42
    // CHECK: regValue = [[MANYTHING]];
    sv.bpassign %regValue, %manyThing : i42

    // CHECK: `ifdef FOO
    sv.ifdef.procedural "FOO" {
      sv.ifdef.procedural "BAR" {
        // Check that the temporary is inserted at the right level, not at the
        // level of the #ifdef.
        %manyMixed = comb.xor %thing, %thing, %thing, %thing, %thing, %thing,
                              %thing, %thing, %thing, %thing, %thing, %thing,
                              %thing, %thing, %thing, %thing, %thing, %thing,
                              %thing, %thing, %thing, %thing, %thing, %thing : i42
        sv.bpassign %regValue, %manyMixed : i42
      }
    }
  }
}

// Issue #2335: https://github.com/llvm/circt/issues/2335
// CHECK-LABEL: module AggregateTemporay(
hw.module @AggregateTemporay(%clock: i1, %foo: i1, %bar: i25) {
  %temp1 = sv.reg  : !hw.inout<!hw.struct<b: i1>>
  %temp2 = sv.reg  : !hw.inout<!hw.array<5x!hw.array<5x!hw.struct<b: i1>>>>
  sv.always posedge %clock  {
    // CHECK: automatic struct packed {logic b; } [[T0:.+]];
    // CHECK: automatic struct packed {logic b; }[4:0][4:0] [[T1:.+]];
    %0 = hw.bitcast %foo : (i1) -> !hw.struct<b: i1>
    sv.passign %temp1, %0 : !hw.struct<b: i1>
    sv.passign %temp1, %0 : !hw.struct<b: i1>
    %1 = hw.bitcast %bar : (i25) -> !hw.array<5x!hw.array<5x!hw.struct<b: i1>>>
    sv.passign %temp2, %1 : !hw.array<5x!hw.array<5x!hw.struct<b: i1>>>
    sv.passign %temp2, %1 : !hw.array<5x!hw.array<5x!hw.struct<b: i1>>>
  }
}

//CHECK-LABEL: module XMR_src
//CHECK: assign $root.a.b.c = a;
//CHECK-NEXT: assign aa = d.e.f;
hw.module @XMR_src(%a : i23) -> (aa: i3) {
  %xmr1 = sv.xmr isRooted a,b,c : !hw.inout<i23>
  %xmr2 = sv.xmr "d",e,f : !hw.inout<i3>
  %r = sv.read_inout %xmr2 : !hw.inout<i3>
  sv.assign %xmr1, %a : i23
  hw.output %r : i3
}

// Test that XMRRefOps are emitted correctly in instances and in binds.  XMRs
// that include verbatim paths and those that do not are both tested.
// Additionally, test that XMRs use properly legalized Verilog names.  The XMR
// target is "new" and the root of the reference is "wait_order".

hw.hierpath private @ref [@wait_order::@bar, @XMRRef_Bar::@new]
hw.hierpath private @ref2 [@wait_order::@baz]
hw.module @XMRRef_Bar() {
  %new = sv.wire sym @new : !hw.inout<i2>
}
hw.module.extern @XMRRef_Baz(%a: i2, %b: i1)
hw.module.extern @XMRRef_Qux(%a: i2, %b: i1)
// CHECK-LABEL: module wait_order
hw.module @wait_order() {
  hw.instance "bar" sym @bar @XMRRef_Bar() -> ()
  %xmr = sv.xmr.ref @ref : !hw.inout<i2>
  %xmrRead = sv.read_inout %xmr : !hw.inout<i2>
  %xmr2 = sv.xmr.ref @ref2 ".x.y.z[42]" : !hw.inout<i1>
  %xmr2Read = sv.read_inout %xmr2 : !hw.inout<i1>
  // CHECK:      /* This instance is elsewhere emitted as a bind statement.
  // CHECK-NEXT: XMRRef_Baz baz (
  // CHECK-NEXT:   .a (wait_order_0.bar.new_0),
  // CHECK-NEXT:   .b (wait_order_0.baz.x.y.z[42])
  // CHECK-NEXT: );
  // CHECK-NEXT: */
  hw.instance "baz" sym @baz @XMRRef_Baz(a: %xmrRead: i2, b: %xmr2Read: i1) -> () {
    doNotPrint = true
  }
  // CHECK-NEXT: XMRRef_Qux qux (
  // CHECK-NEXT:   .a (wait_order_0.bar.new_0),
  // CHECK-NEXT:   .b (wait_order_0.baz.x.y.z[42])
  // CHECK-NEXT: );
  hw.instance "qux" sym @qux @XMRRef_Qux(a: %xmrRead: i2, b: %xmr2Read: i1) -> ()
}

hw.module.extern @MyExtModule(%in: i8)
hw.module.extern @ExtModule(%in: i8) -> (out: i8)

// CHECK-LABEL: module InlineBind
// CHEC:        output wire_0
hw.module @InlineBind(%a_in: i8) -> (wire: i8){
  // CHECK:      wire [7:0] _ext1_out;
  // CHECK-NEXT: wire [7:0] _GEN;
  // CHECK-NEXT: /* This instance is elsewhere emitted as a bind statement.
  // CHECK-NEXT:   ExtModule ext1 (
  // CHECK-NEXT:     .in  (8'(a_in + _GEN)),
  // CHECK-NEXT:     .out (_ext1_out)
  // CHECK-NEXT:   );
  // CHECK-NEXT: */
  // CHECK-NEXT: /* This instance is elsewhere emitted as a bind statement.
  // CHECK-NEXT:   ExtModule ext2 (
  // CHECK-NEXT:     .in  (_ext1_out),
  // CHECK-NEXT:     .out (wire_0)
  // CHECK-NEXT:   );
  // CHECK-NEXT: */
  %0 = sv.wire : !hw.inout<i8>
  %1 = sv.read_inout %0: !hw.inout<i8>
  %2 = comb.add %a_in, %1 : i8
  %3 = hw.instance "ext1" sym @foo1 @ExtModule(in: %2: i8) -> (out: i8) {doNotPrint=1}
  %4 = hw.instance "ext2" sym @foo2 @ExtModule(in: %3: i8) -> (out: i8) {doNotPrint=1}
  hw.output %4: i8
}

// CHECK-LABEL: module MoveInstances
hw.module @MoveInstances(%a_in: i8) -> (outc : i8){
  // CHECK: MyExtModule xyz3 (
  // CHECK:   .in (8'(a_in + a_in))
  // CHECK: );
  // CHECK: assign outc = 8'((8'(8'(a_in + a_in) + a_in)) * a_in)

  %0 = comb.add %a_in, %a_in : i8
  hw.instance "xyz3" @MyExtModule(in: %0: i8) -> ()
  %1 = comb.add %a_in, %a_in : i8
  %2 = comb.add %1, %a_in : i8
  %outc = comb.mul %2, %a_in : i8
  hw.output %outc : i8
}

// CHECK-LABEL: module extInst
hw.module.extern @extInst(%_h: i1, %_i: i1, %_j: i1, %_k: i1, %_z :i0) -> ()

// CHECK-LABEL: module extInst2
// CHECK-NEXT:     input                signed_0,
// CHECK-NEXT:                          _i,
// CHECK-NEXT:                          _j,
// CHECK-NEXT:                          _k
hw.module @extInst2(%signed: i1, %_i: i1, %_j: i1, %_k: i1, %_z :i0) -> () {}

// CHECK-LABEL: module zeroWidthArrayIndex
hw.module @zeroWidthArrayIndex(%clock : i1, %data : i64) -> () {
  %reg = sv.reg  : !hw.inout<uarray<1xi64>>
  sv.alwaysff(posedge %clock) {
    %c0_i0_1 = hw.constant 0 : i0
    // CHECK: reg_0[/*Zero width*/ 1'b0] <= data;
    %0 = sv.array_index_inout %reg[%c0_i0_1] : !hw.inout<uarray<1xi64>>, i0
    sv.passign %0, %data : i64
  }
}

// CHECK-LABEL: module remoteInstDut
hw.module @remoteInstDut(%i: i1, %j: i1, %z: i0) -> () {
  %mywire = sv.wire : !hw.inout<i1>
  %mywire_rd = sv.read_inout %mywire : !hw.inout<i1>
  %myreg = sv.reg : !hw.inout<i1>
  %myreg_rd = sv.read_inout %myreg : !hw.inout<i1>
  %signed = sv.wire  : !hw.inout<i1>
  %mywire_rd1 = sv.read_inout %signed : !hw.inout<i1>
  %output = sv.reg : !hw.inout<i1>
  %myreg_rd1 = sv.read_inout %output: !hw.inout<i1>
  %0 = hw.constant 1 : i1
  hw.instance "a1" sym @bindInst @extInst(_h: %mywire_rd: i1, _i: %myreg_rd: i1, _j: %j: i1, _k: %0: i1, _z: %z: i0) -> () {doNotPrint=1}
  hw.instance "a2" sym @bindInst2 @extInst(_h: %mywire_rd: i1, _i: %myreg_rd: i1, _j: %j: i1, _k: %0: i1, _z: %z: i0) -> () {doNotPrint=1}
  hw.instance "signed" sym @bindInst3 @extInst2(signed: %mywire_rd1 : i1, _i: %myreg_rd1 : i1, _j: %j: i1, _k: %0: i1, _z: %z: i0) -> () {doNotPrint=1}
// CHECK:      wire mywire
// CHECK-NEXT: myreg
// CHECK-NEXT: wire signed_0
// CHECK-NEXT: reg  output_0
// CHECK-NEXT: /* This instance is elsewhere emitted as a bind statement
// CHECK-NEXT:    extInst a1
// CHECK: /* This instance is elsewhere emitted as a bind statement
// CHECK-NEXT:    extInst a2
// CHECK:      /* This instance is elsewhere emitted as a bind statement
// CHECK-NEXT:    extInst2 signed_1
// CHECK-NEXT:    .signed_0 (signed_0)
}

// CHECK-LABEL: SimplyNestedElseIf
// CHECK: if (flag1)
// CHECK: else if (flag2)
// CHECK: else if (flag3) begin
// CHECK-NEXT: (* sv attr *)
// CHECK-NEXT: if (flag4)
// CHECK: end
// CHECK: else
hw.module @SimplyNestedElseIf(%clock: i1, %flag1 : i1, %flag2: i1, %flag3: i1, %flag4: i1) {
  %fd = hw.constant 0x80000002 : i32

  sv.always posedge %clock {
    sv.if %flag1 {
      sv.fwrite %fd, "A"
    } else {
      sv.if %flag2 {
        sv.fwrite %fd, "B"
      } else {
        sv.if %flag3 {
          sv.if %flag4 {
            sv.fwrite %fd, "E"
          } {sv.attributes = [#sv.attribute<"sv attr">]}
          sv.fwrite %fd, "C"
        } else {
          sv.fwrite %fd, "D"
        }
      }
    }
  }

  hw.output
}

// CHECK-LABEL: DoNotChainElseIf
// CHECK: if (flag1)
// CHECK: else begin
// CHECK: if (flag2)
// CHECK: else
// CHECK: end
hw.module @DoNotChainElseIf(%clock: i1, %flag1 : i1, %flag2: i1) {
  %wire = sv.reg : !hw.inout<i32>
  %fd = hw.constant 0x80000002 : i32

  sv.always posedge %clock {
    sv.if %flag1 {
      sv.fwrite %fd, "A"
    } else {
      sv.passign %wire, %fd : i32
      sv.if %flag2 {
        sv.fwrite %fd, "B"
      } else {
        sv.fwrite %fd, "C"
      }
    }
  }

  hw.output
}

// CHECK-LABEL: NestedElseIfHoist
// CHECK: if (flag1)
// CHECK: else begin
// CHECK: automatic logic _GEN;
// CHECK: _GEN = flag2 & flag4;
hw.module @NestedElseIfHoist(%clock: i1, %flag1 : i1, %flag2: i1, %flag3: i1, %flag4 : i1, %arg0: i32, %arg1: i32, %arg2: i32) {
  %fd = hw.constant 0x80000002 : i32

  sv.always posedge %clock {
    sv.if %flag1 {
      sv.fwrite %fd, "A"
    } else {
      %0 = comb.and %flag2, %flag4 : i1
      %10 = comb.or %arg0, %arg1 : i32
      sv.if %0 {
        sv.fwrite %fd, "B"
      } else {
        %1 = comb.and %flag3, %0 : i1
        %11 = comb.or %10, %arg2 : i32
        sv.if %1 {
          sv.fwrite %fd, "C"
        } else {
          sv.fwrite %fd, "D(%d)" (%11) : i32
        }
      }
    }
  }

  hw.output
}

// CHECK-LABEL: ElseIfLocations
// CHECK: if (~flag1)
// CHECK-SAME: // Flag:1:1, If:1:1
// CHECK: else if (~flag2)
// CHECK-SAME: // Flag:2:2, If:2:2
// CHECK: else if (~flag3)
// CHECK-SAME: // Flag:3:3, If:3:3
// CHECK: else
// CHECK-SAME: // If:3:3
hw.module @ElseIfLocations(%clock: i1, %flag1 : i1, %flag2: i1, %flag3: i1) {
  %fd = hw.constant 0x80000002 : i32
  %true = hw.constant 1 : i1

  sv.always posedge %clock {
    %0 = comb.xor %flag1, %true : i1 loc("Flag":1:1)
    sv.if %0 {
      sv.fwrite %fd, "A"
    } else {
      %1 = comb.xor %flag2, %true : i1 loc("Flag":2:2)
      sv.if %1 {
        sv.fwrite %fd, "B"
      } else {
        %2 = comb.xor %flag3, %true : i1 loc("Flag":3:3)
        sv.if %2 {
          sv.fwrite %fd, "C"
        } else {
          sv.fwrite %fd, "D"
        } loc("If":3:3)
      } loc("If":2:2)
    } loc("If":1:1)
  }

  hw.output
}

// CHECK-LABEL: ReuseExistingInOut
// CHECK: input {{.+}}, //
// CHECK:        [[INPUT:[:alnum:]+]], //
// CHECK: output [[OUTPUT:.+]] //
// CHECK: );
hw.module @ReuseExistingInOut(%clock: i1, %a: i1) -> (out1: i1) {
  %expr1 = comb.or %a, %a : i1
  %expr2 = comb.and %a, %a : i1

  // CHECK: wire [[WIRE1:.+]];
  // CHECK: wire [[WIRE2:.+]];
  // CHECK: reg  [[REG:.+]];
  %mywire = sv.wire : !hw.inout<i1>
  %otherwire = sv.wire : !hw.inout<i1>
  %myreg = sv.reg : !hw.inout<i1>

  // CHECK: assign [[WIRE1]] = [[INPUT]] | [[INPUT]];
  sv.assign %mywire, %expr1 : i1

  sv.always posedge %clock {
    // CHECK: [[REG]] <= [[WIRE1]];
    sv.passign %myreg, %expr1 : i1
  }

  %0 = comb.or %a, %expr2 : i1

  // CHECK: assign [[WIRE2]] = [[INPUT]] & [[INPUT]];
  sv.assign %otherwire, %expr2 : i1

  // CHECK: assign [[OUTPUT]] = [[INPUT]] | [[WIRE2]];
  hw.output %0 : i1
}

// CHECK-LABEL: ProhibitReuseOfExistingInOut
hw.module @ProhibitReuseOfExistingInOut(%a: i1) -> (out1: i1) {
  // CHECK-DAG:   wire [[GEN:.+]] = a | a;
  // CHECK-DAG:   wire mywire;
  // CHECK:       `ifdef FOO
  // CHECK-NEXT:    assign mywire = [[GEN]];
  // CHECK-NEXT: `endif
  // CHECK-NEXT: assign out1 = [[GEN]];
  %0 = comb.or %a, %a : i1
  %mywire = sv.wire  : !hw.inout<i1>
  sv.ifdef "FOO" {
    sv.assign %mywire, %0 : i1
  }
  hw.output %0 : i1
}

// See https://github.com/verilator/verilator/issues/3405.
// CHECK-LABEL: Verilator3405
// CHECK-DAG: wire [[GEN0:.+]] =
// CHECK-DAG-NEXT: {{.+}} | {{.+}} | {{.+}}
// CHECK-DAG: wire [[GEN1:.+]] =
// CHECK-DAG-NEXT: {{.+}} | {{.+}} | {{.+}}

// CHECK: assign out = {[[GEN0]], [[GEN1]]}
hw.module @Verilator3405(
  %0: i1, %1: i1, %2: i1, %3: i1, %4: i1, %5: i1, %6: i1, %7: i1, %8: i1,
  %9: i1, %10: i1, %11: i1, %12: i1, %13: i1, %14: i1, %15: i1, %16: i1,
  %17: i1, %18: i1, %19: i1, %20: i1, %21: i1, %22: i1) -> (out: i2) {

  %lhs = comb.or %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10 : i1
  %rhs = comb.or %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21 : i1

  %out = comb.concat %lhs, %rhs : i1, i1

  hw.output %out : i2
}

hw.module @prohiditInline(%a:i4, %b:i1, %c:i1, %d: i4) {
  %0 = sv.reg : !hw.inout<i4>
  %1 = sv.reg : !hw.inout<i4>
  %2 = sv.reg : !hw.inout<i2>
  sv.always posedge %b {
    // CHECK: automatic logic [3:0][[GEN1:.+]];
    // CHECK: [[GEN2:_GEN.*]] = a;
    sv.bpassign %0, %a: i4
    // CHECK-NEXT: [[GEN1]] = 4'([[GEN2]] + d);
    %3 = sv.read_inout %0: !hw.inout<i4>
    %add =comb.add %3, %d: i4
    %extract = comb.extract %add from 1 : (i4) -> i2
    sv.bpassign %2, %extract: i2
  }
}

// CHECK-LABEL: module CollectNamesOrder
hw.module @CollectNamesOrder(%in: i1) -> (out: i1) {
  // CHECK: wire _GEN_0 = in | in;
  // CHECK: wire _GEN;
  %0 = comb.or %in, %in : i1
  %1 = comb.or %0, %0 : i1
  %foo = sv.wire {hw.verilogName = "_GEN" } : !hw.inout<i1>
  hw.output %1 : i1
}

// CHECK-LABEL: module InlineReadInout
hw.module private @InlineReadInout() -> () {
  %c0_i32 = hw.constant 0 : i32
  %false = hw.constant false
  %r1 = sv.reg  : !hw.inout<i2>
  sv.initial {
    %_RANDOM = sv.logic  : !hw.inout<uarray<1xi32>>
    %2 = sv.array_index_inout %_RANDOM[%c0_i32] : !hw.inout<uarray<1xi32>>, i32
    %RAMDOM = sv.verbatim.expr.se "`RAMDOM" : () -> i32 {symbols = []}
    sv.bpassign %2, %RAMDOM : i32
    %3 = sv.array_index_inout %_RANDOM[%c0_i32] : !hw.inout<uarray<1xi32>>, i32
    %4 = sv.read_inout %3 : !hw.inout<i32>
    // CHECK: automatic logic [31:0] _RANDOM[0:0];
    // CHECK: _RANDOM[32'h0] = `RAMDOM;
    // CHECK-NEXT: r1 = _RANDOM[32'h0][1:0];
    %5 = comb.extract %4 from 0 : (i32) -> i2
    sv.bpassign %r1, %5 : i2
  }
}

// CHECK-LABEL: module Dollar
hw.module private @Dollar(%cond: i1) -> () {
  sv.initial {
    // CHECK:      _$a:
    // CHECK-NEXT: a$:
    sv.cover %cond, immediate label "$a"
    sv.cover %cond, immediate label "a$"
  }
}

// CHECK-LABEL: IndexPartSelectInoutArray
hw.module @IndexPartSelectInoutArray(%a: !hw.array<2xi1>, %c: i1, %d: i1) {
  %c0_i2 = hw.constant 0 : i2
  %r1 = sv.reg  : !hw.inout<array<3xi1>>
  sv.always posedge %d {
    // CHECK: r1[2'h0 +: 2] <= a;
    %1 = sv.indexed_part_select_inout %r1[%c0_i2 : 2] : !hw.inout<array<3xi1>>, i2
    sv.passign %1, %a : !hw.array<2xi1>
  }
  hw.output
}

hw.module @IndexPartSelect() -> (a : i3) {
  // CHECK-LABEL: module IndexPartSelect(
  // CHECK: wire [17:0] _GEN = 18'h3;
  // CHECK-NEXT: assign a = _GEN[4'h3 +: 3];
  %c3_i3 = hw.constant 3 : i4
  %c3_i18 = hw.constant 3 : i18
  %c = sv.indexed_part_select %c3_i18[%c3_i3 : 3] : i18,i4
  hw.output %c : i3
}

// CHECK-LABEL: module ConditionalComments(
hw.module @ConditionalComments() {
  sv.ifdef "FOO"  {             // CHECK-NEXT: `ifdef FOO
    sv.verbatim "`define FOO_A" // CHECK-NEXT:   `define FOO_A
  } else  {                     // CHECK-NEXT: `else  // FOO
    sv.verbatim "`define FOO_B" // CHECK-NEXT:   `define FOO_B
  }                             // CHECK-NEXT: `endif // FOO

  sv.ifdef "BAR"  {             // CHECK-NEXT: `ifndef BAR
  } else  {
    sv.verbatim "`define X"     // CHECK-NEXT:   `define X
  }                             // CHECK-NEXT: `endif // not def BAR
}

sv.macro.decl @RANDOM
sv.macro.decl @PRINTF_COND_

// CHECK-LABEL: module ForStatement
hw.module @ForStatement(%a: i5) -> () {
  %_RANDOM = sv.logic : !hw.inout<uarray<3xi32>>
  sv.initial {
    %c-2_i2 = hw.constant -2 : i2
    %c1_i2 = hw.constant 1 : i2
    %c-1_i2 = hw.constant -1 : i2
    %c0_i2 = hw.constant 0 : i2
    // CHECK:      for (logic [1:0] i = 2'h0; i < 2'h3; i += 2'h1) begin
    // CHECK-NEXT:   _RANDOM[i] = `RANDOM;
    // CHECK-NEXT: end
    sv.for %i = %c0_i2 to %c-1_i2 step %c1_i2 : i2 {
      %RANDOM = sv.macro.ref.se @RANDOM() : ()->i32
      %index = sv.array_index_inout %_RANDOM[%i] : !hw.inout<uarray<3xi32>>, i2
      sv.bpassign %index, %RANDOM : i32
    }
  }
}

// CHECK-LABEL: module intrinsic
hw.module @intrinsic(%clk: i1) -> (io1: i1, io2: i1, io3: i1, io4: i5) {
  // CHECK: wire [4:0] [[tmp:.*]];

  %x_i1 = sv.constantX : i1
  %0 = comb.icmp bin ceq %clk, %x_i1 : i1
  // CHECK: assign io1 = clk === 1'bx

  %1 = sv.constantStr "foo"
  %2 = sv.system "test$plusargs"(%1) : (!hw.string) -> i1
  // CHECK: assign io2 = $test$plusargs("foo")

  %_pargs = sv.wire  : !hw.inout<i5>
  %3 = sv.read_inout %_pargs : !hw.inout<i5>
  %4 = sv.system "value$plusargs"(%1, %_pargs) : (!hw.string, !hw.inout<i5>) -> i1
  // CHECK: assign io3 = $value$plusargs("foo", [[tmp]])
  // CHECK: assign io4 = [[tmp]]

  hw.output %0, %2, %4, %3 : i1, i1, i1, i5
}

hw.module @bindInMod() {
  sv.bind #hw.innerNameRef<@remoteInstDut::@bindInst>
  sv.bind #hw.innerNameRef<@remoteInstDut::@bindInst3>
}

// CHECK-LABEL: module bindInMod();
// CHECK-NEXT:   bind remoteInstDut extInst a1 (
// CHECK-NEXT:   ._h (mywire),
// CHECK-NEXT:   ._i (myreg),
// CHECK-NEXT:   ._j (j),
// CHECK-NEXT:   ._k (1'h1)
// CHECK-NEXT: //._z (z)
// CHECK-NEXT: );
// CHECK-NEXT:  bind remoteInstDut extInst2 signed_1 (
// CHECK-NEXT:    .signed_0 (signed_0),
// CHECK-NEXT:    ._i       (output_0),
// CHECK-NEXT:    ._j       (j),
// CHECK-NEXT:    ._k       (1'h1)
// CHECK: endmodule

sv.bind <@wait_order::@baz>

// CHECK-LABEL: bind wait_order_0 XMRRef_Baz baz (
// CHECK-NEXT:    .a (wait_order_0.bar.new_0)
// CHECK-NEXT:    .b (wait_order_0.baz.x.y.z[42])
// CHECK-NEXT:  );

sv.bind #hw.innerNameRef<@remoteInstDut::@bindInst2>

// CHECK-LABEL: bind remoteInstDut extInst a2 (
// CHECK-NEXT:   ._h (mywire),
// CHECK-NEXT:   ._i (myreg),
// CHECK-NEXT:   ._j (j),
// CHECK-NEXT:   ._k (1'h1)
// CHECK-NEXT: //._z (z)
// CHECK-NEXT: );

// Regression test for a bug where bind emission would not use sanitized names.
hw.module @NastyPortParent() {
  %false = hw.constant false
  %0 = hw.instance "foo" sym @foo @NastyPort(".lots$of.dots": %false: i1) -> (".more.dots": i1) {doNotPrint = true}
}
hw.module @NastyPort(%.lots$of.dots: i1) -> (".more.dots": i1) {
  %false = hw.constant false
  hw.output %false : i1
}
sv.bind #hw.innerNameRef<@NastyPortParent::@foo>
// CHECK-LABEL: bind NastyPortParent NastyPort foo (
// CHECK-NEXT:    ._lots$of_dots (1'h0)
// CHECK-NEXT:    ._more_dots     (/* unused */)
// CHECK-NEXT:  );

sv.bind #hw.innerNameRef<@InlineBind::@foo1>
sv.bind #hw.innerNameRef<@InlineBind::@foo2>
// CHECK-LABEL: bind InlineBind ExtModule ext1 (
// CHECK-NEXT:    .in  (8'(a_in + _GEN))
// CHECK-NEXT:    .out (_ext1_out)
// CHECK-NEXT:  );
// CHECK-LABEL: bind InlineBind ExtModule ext2 (
// CHECK-NEXT:    .in  (_ext1_out)
// CHECK-NEXT:    .out (wire_0)
// CHECK-NEXT:  );

// CHECK-LABEL:  hw.module @issue595
// CHECK:     sv.wire  {hw.verilogName = "_GEN"} : !hw.inout<i32>

// CHECK-LABEL: hw.module @extInst2
// CHECK-SAME: (%signed: i1 {hw.verilogName = "signed_0"}

// CHECK-LABEL:  hw.module @remoteInstDut
// CHECK:    %signed = sv.wire  {hw.verilogName = "signed_0"} : !hw.inout<i1>
// CHECK:    %output = sv.reg  {hw.verilogName = "output_0"} : !hw.inout<i1>
