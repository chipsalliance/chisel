// RUN: circt-opt %s -test-apply-lowering-options='options=emitBindComments' -export-verilog -verify-diagnostics | FileCheck %s --strict-whitespace


hw.testmodule @NewStyle (input %a : i3, 
                         output %b : i3, 
                         input %c : i4, 
                         output %d : i4, 
                         inout %e : i64 {hw.exportPort = #hw<innerSym@symA>}) {
  hw.output %a, %c : i3, i4
 }

// CHECK-LABEL: module inputs_only(
// CHECK-NEXT:   input a,{{.*}}
// CHECK-NEXT:         b
// CHECK-NEXT:  );
hw.module @inputs_only(%a: i1, %b: i1) {
  hw.output
}

// CHECK-LABEL: module no_ports();
hw.module @no_ports() {
  hw.output
}

// CHECK-LABEL: module Expressions(
// CHECK-NEXT:    input  [3:0]  in4,
// CHECK-NEXT:    input         clock,
// CHECK-NEXT:    output        out1a,
// CHECK-NEXT:                  out1b,
// CHECK-NEXT:                  out1c,
// CHECK-NEXT:                  out1d,
// CHECK-NEXT:                  out1e,
// CHECK-NEXT:                  out1f,
// CHECK-NEXT:                  out1g,
// CHECK-NEXT:    output [3:0]  out4,
// CHECK-NEXT:                  out4s,
// CHECK-NEXT:    output [15:0] out16,
// CHECK-NEXT:                  out16s,
// CHECK-NEXT:    output [16:0] sext17,
// CHECK-NEXT:    output [1:0]  orvout
// CHECK-NEXT:  );

hw.module @Expressions(%in4: i4, %clock: i1) ->
  (out1a: i1, out1b: i1, out1c: i1, out1d: i1, out1e: i1, out1f: i1, out1g: i1,
   out4: i4, out4s: i4, out16: i16, out16s: i16, sext17: i17, orvout: i2) {
  %c1_i4 = hw.constant 1 : i4
  %c2_i4 = hw.constant 2 : i4
  %c3_i4 = hw.constant 3 : i4
  %c-1_i4 = hw.constant -1 : i4
  %c0_i4 = hw.constant 0 : i4
  %false = hw.constant false
  %c0_i2 = hw.constant 0 : i2
  %c0_i5 = hw.constant 0 : i5
  %c0_i6 = hw.constant 0 : i6
  %c0_i10 = hw.constant 0 : i10

  // CHECK: wire [3:0]  _GEN = in4 >> in4;
  %7 = comb.extract %in4 from 2 : (i4) -> i1

  %10 = comb.shru %in4, %in4 : i4

  // CHECK: assign w1 = ~in4;
  %3 = comb.xor %in4, %c-1_i4 : i4

  // CHECK: assign w1 = in4 % 4'h1;
  %4 = comb.modu %in4, %c1_i4 : i4

  // CHECK: assign w1 = {2'h0, in4[1:0]};
  %5 = comb.extract %in4 from 0 : (i4) -> i2

  // CHECK: assign w1 = {2'h0, in4[3:2] | {in4[2], 1'h0}};
  %6 = comb.extract %in4 from 2 : (i4) -> i2
  %8 = comb.concat %7, %false : i1, i1
  %9 = comb.or %6, %8 : i2

  // CHECK: assign w1 = _GEN;
  // CHECK: assign w1 = clock ? (clock ? 4'h1 : 4'h2) : 4'h3;
  // CHECK: assign w1 = clock ? 4'h1 : clock ? 4'h2 : 4'h3;
  %11 = comb.shrs %in4, %in4 : i4
  %12 = comb.concat %false, %in4, %in4 : i1, i4, i4
  %13 = comb.mux %clock, %c1_i4, %c2_i4 : i4
  %14 = comb.mux %clock, %13, %c3_i4 : i4
  %15 = comb.mux %clock, %c2_i4, %c3_i4 : i4
  %16 = comb.mux %clock, %c1_i4, %15 : i4

  // CHECK: assign w1 = {2'h0, in4[3:2] | in4[1:0]};
  %17 = comb.or %6, %5 : i2
  %18 = comb.concat %c0_i2, %in4 : i2, i4

  // CHECK: assign w2 = {6'h0, in4, clock, clock, in4};
  // CHECK: assign w2 = {10'h0, {2'h0, in4} ^ {{..}}2{in4[3]}}, in4} ^ {6{clock}}};
  %tmp = comb.extract %in4 from 3 : (i4) -> i1
  %tmp2 = comb.replicate %tmp : (i1) -> i2
  %19 = comb.concat %tmp2, %in4 : i2, i4
  %20 = comb.replicate %clock : (i1) -> i6
  %21 = comb.xor %18, %19, %20 : i6
  %tmp3 = comb.extract %in4 from 3 : (i4) -> i1
  %22 = comb.concat %tmp3, %in4 : i1, i4
  %23 = comb.sub %c0_i5, %22 : i5
  %25 = comb.concat %c0_i2, %5 : i2, i2
  %26 = comb.concat %c0_i2, %9 : i2, i2
  %27 = comb.concat %c0_i2, %17 : i2, i2

  %w1 = sv.wire : !hw.inout<i4>
  %w1_use = sv.read_inout %w1 : !hw.inout<i4>

  sv.assign %w1, %3 : i4
  sv.assign %w1, %4 : i4
  sv.assign %w1, %25 : i4
  sv.assign %w1, %26 : i4
  sv.assign %w1, %10 : i4
  sv.assign %w1, %14 : i4
  sv.assign %w1, %16 : i4
  sv.assign %w1, %27 : i4
  sv.assign %w1, %10 : i4

  %29 = comb.concat %c0_i6, %in4, %clock, %clock, %in4 : i6, i4, i1, i1, i4
  %30 = comb.concat %c0_i10, %21 : i10, i6

  %w2 = sv.wire : !hw.inout<i16>
  %w2_use = sv.read_inout %w2 : !hw.inout<i16>
  sv.assign %w2, %29 : i16
  sv.assign %w2, %30 : i16

  %w3 = sv.wire : !hw.inout<i16>
  %w3_use = sv.read_inout %w3 : !hw.inout<i16>


 // CHECK: assign out1a = ^in4;
  %0 = comb.parity %in4 : i4
  // CHECK: assign out1b = &in4;
  %1 = comb.icmp eq %in4, %c-1_i4 : i4
  // CHECK: assign out1c = |in4;
  %2 = comb.icmp ne %in4, %c0_i4 : i4

  // CHECK: assign out1d = in4 === 4'h0;
  %cmp3 = comb.icmp ceq %in4, %c0_i4 : i4
  // CHECK: assign out1e = in4 !== 4'h0;
  %cmp4 = comb.icmp cne %in4, %c0_i4 : i4
  // CHECK: assign out1f = in4 ==? 4'h0;
  %cmp5 = comb.icmp weq %in4, %c0_i4 : i4
  // CHECK: assign out1g = in4 !=? 4'h0;
  %cmp6 = comb.icmp wne %in4, %c0_i4 : i4

  // CHECK: assign out4s = $signed($signed(in4) >>> in4);
  // CHECK: assign sext17 = {w3[15], w3};
  %36 = comb.extract %w3_use from 15 : (i16) -> i1
  %35 = comb.concat %36, %w3_use : i1, i16

  // Variadic with name attribute lowers
  // CHECK: assign orvout = in4[1:0] | in4[3:2] | in4[2:1];
  %orpre1 = comb.extract %in4 from 0 : (i4) -> i2
  %orpre2 = comb.extract %in4 from 2 : (i4) -> i2
  %orpre3 = comb.extract %in4 from 1 : (i4) -> i2
  %orv = comb.or %orpre1, %orpre2, %orpre3 {sv.namehint = "hintyhint"}: i2
  hw.output %0, %1, %2, %cmp3, %cmp4, %cmp5, %cmp6, %w1_use, %11, %w2_use, %w3_use, %35, %orv : i1, i1, i1, i1, i1, i1, i1, i4, i4, i16, i16, i17, i2
}

// CHECK-LABEL: module Precedence(
hw.module @Precedence(%a: i4, %b: i4, %c: i4) -> (out1: i1, out: i10) {
  %false = hw.constant false
  %c0_i2 = hw.constant 0 : i2
  %c0_i4 = hw.constant 0 : i4
  %c0_i5 = hw.constant 0 : i5
  %c0_i3 = hw.constant 0 : i3
  %c0_i6 = hw.constant 0 : i6
  %_out1_output = sv.wire  : !hw.inout<i1>
  %_out_output = sv.wire  : !hw.inout<i10>

  // CHECK: wire [4:0] _GEN = {1'h0, b};
  // CHECK: wire [4:0] _GEN_0 = _GEN + {1'h0, c};
  // CHECK: wire [5:0] _GEN_1 = {2'h0, a};
  // CHECK: wire [5:0] _GEN_2 = {1'h0, _GEN_0};
  // CHECK: assign _out_output = {4'h0, _GEN_1 + _GEN_2};
  // CHECK: wire [4:0] _GEN_3 = {1'h0, a} + _GEN;
  // CHECK: assign _out_output = {4'h0, {1'h0, _GEN_3} - {2'h0, c}};
  // CHECK: assign _out_output = {4'h0, _GEN_1 - _GEN_2};
  // CHECK: wire [7:0] _GEN_4 = {4'h0, b};
  // CHECK: wire [8:0] _GEN_5 = {5'h0, a};
  // CHECK: assign _out_output = {1'h0, _GEN_5 + {1'h0, _GEN_4 * {4'h0, c}}};
  // CHECK: wire [8:0] _GEN_6 = {5'h0, c};
  // CHECK: assign _out_output = {1'h0, {1'h0, {4'h0, a} * _GEN_4} + _GEN_6};
  // CHECK: assign _out_output = {1'h0, {4'h0, _GEN_3} * _GEN_6};
  // CHECK: assign _out_output = {1'h0, _GEN_5 * {4'h0, _GEN_0}};
  // CHECK: assign _out_output = {5'h0, _GEN_3} * {5'h0, _GEN_0};
  // CHECK: assign _out1_output = ^_GEN_0;
  // CHECK: assign _out1_output = b < c | b > c;
  // CHECK: assign _out_output = {6'h0, (b ^ c) & {3'h0, _out1_output}};
  // CHECK: assign _out_output = {2'h0, _out_output[9:2]};
  // CHECK: assign _out1_output = _out_output < {6'h0, a};
  %0 = comb.concat %false, %b : i1, i4
  %1 = comb.concat %false, %c : i1, i4
  %2 = comb.add %0, %1 : i5
  %3 = comb.concat %c0_i2, %a : i2, i4
  %4 = comb.concat %false, %2 : i1, i5
  %5 = comb.add %3, %4 : i6
  %6 = comb.concat %c0_i4, %5 : i4, i6
  sv.assign %_out_output, %6 : i10
  %7 = comb.concat %false, %a : i1, i4
  %8 = comb.add %7, %0 : i5
  %9 = comb.concat %false, %8 : i1, i5
  %10 = comb.concat %c0_i2, %c : i2, i4
  %11 = comb.sub %9, %10 : i6
  %12 = comb.concat %c0_i4, %11 : i4, i6
  sv.assign %_out_output, %12 : i10
  %13 = comb.sub %3, %4 : i6
  %14 = comb.concat %c0_i4, %13 : i4, i6
  sv.assign %_out_output, %14 : i10
  %15 = comb.concat %c0_i4, %b : i4, i4
  %16 = comb.concat %c0_i4, %c : i4, i4
  %17 = comb.mul %15, %16 : i8
  %18 = comb.concat %c0_i5, %a : i5, i4
  %19 = comb.concat %false, %17 : i1, i8
  %20 = comb.add %18, %19 : i9
  %21 = comb.concat %false, %20 : i1, i9
  sv.assign %_out_output, %21 : i10
  %22 = comb.concat %c0_i4, %a : i4, i4
  %23 = comb.mul %22, %15 : i8
  %24 = comb.concat %false, %23 : i1, i8
  %25 = comb.concat %c0_i5, %c : i5, i4
  %26 = comb.add %24, %25 : i9
  %27 = comb.concat %false, %26 : i1, i9
  sv.assign %_out_output, %27 : i10
  %28 = comb.concat %c0_i4, %8 : i4, i5
  %29 = comb.mul %28, %25 : i9
  %30 = comb.concat %false, %29 : i1, i9
  sv.assign %_out_output, %30 : i10
  %31 = comb.concat %c0_i4, %2 : i4, i5
  %32 = comb.mul %18, %31 : i9
  %33 = comb.concat %false, %32 : i1, i9
  sv.assign %_out_output, %33 : i10
  %34 = comb.concat %c0_i5, %8 : i5, i5
  %35 = comb.concat %c0_i5, %2 : i5, i5
  %36 = comb.mul %34, %35 : i10
  sv.assign %_out_output, %36 : i10
  %37 = comb.parity %2 : i5
  sv.assign %_out1_output, %37 : i1
  %38 = comb.icmp ult %b, %c : i4
  %39 = comb.icmp ugt %b, %c : i4
  %40 = comb.or %38, %39 : i1
  sv.assign %_out1_output, %40 : i1
  %41 = comb.xor %b, %c : i4
  %42 = sv.read_inout %_out1_output : !hw.inout<i1>
  %43 = comb.concat %c0_i3, %42 : i3, i1
  %44 = comb.and %41, %43 : i4
  %45 = comb.concat %c0_i6, %44 : i6, i4
  sv.assign %_out_output, %45 : i10
  %46 = sv.read_inout %_out_output : !hw.inout<i10>
  %47 = comb.extract %46 from 2 : (i10) -> i8
  %48 = comb.concat %c0_i2, %47 : i2, i8
  sv.assign %_out_output, %48 : i10
  %49 = comb.concat %c0_i6, %a : i6, i4
  %50 = comb.icmp ult %46, %49 : i10
  sv.assign %_out1_output, %50 : i1
  hw.output %42, %46 : i1, i10
}

// CHECK-LABEL: module CmpSign(
hw.module @CmpSign(%a: i4, %b: i4, %c: i4, %d: i4) ->
 (o0: i1, o1: i1, o2: i1, o3: i1, o4: i1, o5: i1, o6: i1, o7: i1,
  o8: i1, o9: i1, o10: i1, o11: i1, o12: i1, o13: i1, o14: i1, o15: i1) {
  // CHECK: assign o0 = a < b;
  %0 = comb.icmp ult %a, %b : i4
  // CHECK-NEXT: assign o1 = $signed(c) < $signed(d);
  %1 = comb.icmp slt %c, %d : i4
  // CHECK-NEXT: assign o2 = $signed(a) < $signed(b);
  %2 = comb.icmp slt %a, %b : i4
  // CHECK-NEXT: assign o3 = a <= b;
  %3 = comb.icmp ule %a, %b : i4
  // CHECK-NEXT: assign o4 = $signed(c) <= $signed(d);
  %4 = comb.icmp sle %c, %d : i4
  // CHECK-NEXT: assign o5 = $signed(a) <= $signed(b);
  %5 = comb.icmp sle %a, %b : i4
  // CHECK-NEXT: assign o6 = a > b;
  %6 = comb.icmp ugt %a, %b : i4
  // CHECK-NEXT: assign o7 = $signed(c) > $signed(d);
  %7 = comb.icmp sgt %c, %d : i4
  // CHECK-NEXT: assign o8 = $signed(a) > $signed(b);
  %8 = comb.icmp sgt %a, %b : i4
  // CHECK-NEXT: assign o9 = a >= b;
  %9 = comb.icmp uge %a, %b : i4
  // CHECK-NEXT: assign o10 = $signed(c) >= $signed(d);
  // CHECK-NEXT: assign o11 = $signed(a) >= $signed(b);
  %10 = comb.icmp sge %c, %d : i4
  %11 = comb.icmp sge %a, %b : i4
  // CHECK-NEXT: assign o12 = a == b;
  %12 = comb.icmp eq %a, %b : i4
  // CHECK-NEXT: assign o13 = c == d;
  %13 = comb.icmp eq %c, %d : i4
  // CHECK-NEXT: assign o14 = a != b;
  %14 = comb.icmp ne %a, %b : i4
  // CHECK-NEXT: assign o15 = c != d;
  %15 = comb.icmp ne %c, %d : i4

  hw.output %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15 : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
}

// CHECK-LABEL: module Wires(
hw.hierpath @myWirePath [@Wires::@myWire]
hw.module @Wires(%a: i4) -> (x: i4, y: i4) {
  // CHECK-DAG: wire [3:0] wire1 = a;
  // CHECK-DAG: assign x = wire1;
  %wire1 = hw.wire %a : i4

  // Use before def
  // CHECK-DAG: wire [3:0] wire2 = wire1;
  // CHECK-DAG: assign y = wire2 * wire2;
  %0 = comb.mul %wire2, %wire2 : i4
  %wire2 = hw.wire %wire1 : i4

  // Nested use before def
  // CHECK-DAG: wire [3:0] wire4 = a;
  sv.always {
    // CHECK-DAG: logic [3:0] wire3 = a;
    %wire3 = hw.wire %a : i4
    // CHECK-DAG: assert(wire3 == wire4);
    %1 = comb.icmp eq %wire3, %wire4 : i4
    sv.assert %1, immediate
  }
  %wire4 = hw.wire %a : i4

  // Inner symbol references
  // CHECK-DAG: wire [3:0] wire5 = a;
  // CHECK-DAG: symRef1(wire5);
  // CHECK-DAG: symRef2(Wires.wire5);
  %wire5 = hw.wire %a sym @myWire : i4
  sv.verbatim "symRef1({{0}});" {symbols = [#hw.innerNameRef<@Wires::@myWire>]}
  %2 = sv.xmr.ref @myWirePath : !hw.inout<i4>
  %3 = sv.read_inout %2 : !hw.inout<i4>
  sv.verbatim "symRef2({{0}});"(%3) : i4

  hw.output %wire1, %0 : i4, i4
}

// CHECK-LABEL: module MultiUseExpr
hw.module @MultiUseExpr(%a: i4) -> (b0: i1, b1: i1, b2: i1, b3: i1, b4: i2) {
  %false = hw.constant false
  %c1_i5 = hw.constant 1 : i5
  %c-1_i5 = hw.constant -1 : i5
  %c-1_i4 = hw.constant -1 : i4

  // CHECK: wire {{ *}}_GEN = ^a;
  %0 = comb.parity %a : i4
  // CHECK-NEXT: wire [4:0] _GEN_0 = {1'h0, a} << 5'h1;
  %1 = comb.concat %false, %a : i1, i4
  %2 = comb.shl %1, %c1_i5 : i5

  // CHECK-NEXT: wire [3:0] _GEN_1 = ~a;
  // CHECK-NEXT: assign b0 = _GEN;
  // CHECK-NEXT: assign b1 = ^_GEN;
  // CHECK-NEXT: assign b2 = &_GEN_0;
  // CHECK-NEXT: assign b3 = ^_GEN_0;
  // CHECK-NEXT: assign b4 = _GEN_1[3:2];
  %3 = comb.parity %0 : i1
  %4 = comb.icmp eq %2, %c-1_i5 : i5
  %5 = comb.parity %2 : i5
  %6 = comb.xor %a, %c-1_i4 : i4
  %7 = comb.extract %6 from 2 : (i4) -> i2
  hw.output %0, %3, %4, %5, %7 : i1, i1, i1, i1, i2
}

// CHECK-LABEL: module SimpleConstPrint(
// CHECK-NEXT:    input  [3:0] in4,
// CHECK-NEXT:    output [3:0] out4
// CHECK-NEXT: );
// CHECK:  wire [3:0] w = 4'h1;
// CHECK:  assign out4 = in4 + 4'h1;
// CHECK-NEXT: endmodule
hw.module @SimpleConstPrint(%in4: i4) -> (out4: i4) {
  %w = sv.wire : !hw.inout<i4>
  %c1_i4 = hw.constant 1 : i4
  sv.assign %w, %c1_i4 : i4
  %1 = comb.add %in4, %c1_i4 : i4
  hw.output %1 : i4
}

// Use constants, don't fold them into wires
// CHECK-LABEL: module SimpleConstPrintReset(
// CHECK:  q <= 4'h1;
hw.module @SimpleConstPrintReset(%clock: i1, %reset: i1, %in4: i4) -> () {
  %w = sv.wire : !hw.inout<i4>
  %q = sv.reg : !hw.inout<i4>
  %c1_i4 = hw.constant 1 : i4
  sv.assign %w, %c1_i4 : i4
  sv.always posedge %clock, posedge %reset {
    sv.if %reset {
        sv.passign %q, %c1_i4 : i4
      } else {
        sv.passign %q, %in4 : i4
      }
    }
    hw.output

}

// CHECK-LABEL: module InlineDeclAssignment
hw.module @InlineDeclAssignment(%a: i1) {
  // CHECK: wire b = a;
  %b = sv.wire : !hw.inout<i1>
  sv.assign %b, %a : i1

  // CHECK: wire c = a + a;
  %0 = comb.add %a, %a : i1
  %c = sv.wire : !hw.inout<i1>
  sv.assign %c, %0 : i1
}

// CHECK-LABEL: module ordered_region
// CHECK-NEXT: input a
// CHECK-NEXT: );
// CHECK-EMPTY:
hw.module @ordered_region(%a: i1) {
  sv.ordered {
    // CHECK-NEXT: `ifdef foo
    sv.ifdef "foo" {
      // CHECK-NEXT: wire_0 = a;
      %wire = sv.wire : !hw.inout<i1>
      sv.assign %wire, %a : i1
    }
    // CHECK-NEXT: `endif
    // CHECK-NEXT: `ifdef bar
    sv.ifdef "bar" {
      // CHECK-NEXT: wire_1 = a;
      %wire = sv.wire : !hw.inout<i1>
      sv.assign %wire, %a : i1
    }
    // CHECK-NEXT: `endif
  }
}


hw.module.extern @MyExtModule(%in: i8) -> (out: i1) attributes {verilogName = "FooExtModule"}
hw.module.extern @AParameterizedExtModule<CFG: none>(%in: i8) -> (out: i1)

// CHECK-LABEL: module ExternMods
hw.module @ExternMods(%a_in: i8) {
  // CHECK: AParameterizedExtModule #(
  // CHECK:   .CFG(FOO)
  // CHECK: ) xyz2
  hw.instance "xyz2" @AParameterizedExtModule<CFG: none = #hw.param.verbatim<"FOO">>(in: %a_in: i8) -> (out: i1)
  // CHECK: AParameterizedExtModule #(
  // CHECK:   .CFG("STRING")
  // CHECK: ) xyz3
  hw.instance "xyz3" @AParameterizedExtModule<CFG: none = #hw.param.verbatim<"\"STRING\"">>(in: %a_in: i8) -> (out: i1)
}

hw.module.extern @MyParameterizedExtModule<DEFAULT: i32, DEPTH: f64, FORMAT: none,
     WIDTH: i8>(%in: i8) -> (out: i1)

// CHECK-LABEL: module UseInstances
hw.module @UseInstances(%a_in: i8) -> (a_out1: i1, a_out2: i1) {
  // CHECK: FooExtModule xyz (
  // CHECK:   .in  (a_in),
  // CHECK:   .out (a_out1)
  // CHECK: );
  // CHECK: MyParameterizedExtModule #(
  // CHECK:   .DEFAULT(0),
  // CHECK:   .DEPTH(3.500000e+00),
  // CHECK:   .FORMAT("xyz_timeout=%d\n"),
  // CHECK:   .WIDTH(32)
  // CHECK: ) xyz2 (
  // CHECK:   .in  (a_in),
  // CHECK:   .out (a_out2)
  // CHECK: );
  %xyz.out = hw.instance "xyz" @MyExtModule(in: %a_in: i8) -> (out: i1)
  %xyz2.out = hw.instance "xyz2" @MyParameterizedExtModule<
     DEFAULT: i32 = 0, DEPTH: f64 = 3.500000e+00, FORMAT: none = "xyz_timeout=%d\0A",
     WIDTH: i8 = 32
  >(in: %a_in: i8) -> (out: i1)
  hw.output %xyz.out, %xyz2.out : i1, i1
}

// Instantiate a parametric module using parameters from its parent module
hw.module.extern @ExternParametricWidth<width: i32>
  (%in: !hw.int<#hw.param.decl.ref<"width">>) -> (out: !hw.int<#hw.param.decl.ref<"width">>)
// CHECK-LABEL: module NestedParameterUsage
hw.module @NestedParameterUsage<param: i32>(
  %in: !hw.int<#hw.param.decl.ref<"param">>) -> (out: !hw.int<#hw.param.decl.ref<"param">>) {
  // CHECK: #(parameter /*integer*/ param) (
  // CHECK: input  [param - 1:0] in,
  // CHECK: output [param - 1:0] out
  // CHECK: );
  // CHECK: ExternParametricWidth #(
  // CHECK:   .width(param)
  // CHECK: ) externWidth (
  // CHECK:   .in  (in),
  // CHECK:   .out (out)
  // CHECK: );
  // CHECK: endmodule
  %externWidth.out = hw.instance "externWidth"
    @ExternParametricWidth<width: i32 = #hw.param.decl.ref<"param">>(
      in: %in : !hw.int<#hw.param.decl.ref<"param">>) -> (out: !hw.int<#hw.param.decl.ref<"param">>)
  hw.output %externWidth.out : !hw.int<#hw.param.decl.ref<"param">>
}

// CHECK-LABEL: module Stop(
hw.module @Stop(%clock: i1, %reset: i1) {
  // CHECK: always @(posedge clock) begin
  // CHECK:   `ifndef SYNTHESIS
  // CHECK:     if (`STOP_COND_ & reset)
  // CHECK:       $fatal;
  // CHECK:   `endif
  // CHECK: end // always @(posedge)
  sv.always posedge %clock  {
    sv.ifdef.procedural "SYNTHESIS"  {
    } else  {
      %0 = sv.verbatim.expr "`STOP_COND_" : () -> i1
      %1 = comb.and %0, %reset : i1
      sv.if %1  {
        sv.fatal 1
      }
    }
  }
  hw.output
}

sv.macro.decl @PRINTF_COND_

// CHECK-LABEL: module Print
hw.module @Print(%clock: i1, %reset: i1, %a: i4, %b: i4) {
  %fd = hw.constant 0x80000002 : i32
  %false = hw.constant false
  %c1_i5 = hw.constant 1 : i5

  // CHECK: always @(posedge clock) begin
  // CHECK:   if ((`PRINTF_COND_) & reset)
  // CHECK:     $fwrite(32'h80000002, "Hi %x %x\n", {1'h0, a} << 5'h1, b);
  // CHECK: end // always @(posedge)
  %0 = comb.concat %false, %a : i1, i4
  %1 = comb.shl %0, %c1_i5 : i5
  sv.always posedge %clock  {
    %2 = sv.macro.ref @PRINTF_COND_() : () -> i1
    %3 = comb.and %2, %reset : i1
    sv.if %3  {
      sv.fwrite %fd, "Hi %x %x\0A"(%1, %b) : i5, i4
    }
  }
  hw.output
}

// CHECK-LABEL: module ReadMem()
hw.module @ReadMem() {
  // CHECK:      reg [31:0] mem[0:7];
  %mem = sv.reg sym @mem : !hw.inout<uarray<8xi32>>
  // CHECK-NEXT: initial begin
  // CHECK-NEXT:   $readmemb("file1.txt", mem);
  // CHECK-NEXT:   $readmemh("file2.txt", mem);
  // CHECK-NEXT: end
  sv.initial {
    sv.readmem %mem, "file1.txt", MemBaseBin : !hw.inout<uarray<8xi32>>
    sv.readmem %mem, "file2.txt", MemBaseHex : !hw.inout<uarray<8xi32>>
  }

}

// CHECK: module ReadMemXMR()
hw.hierpath @ReadMemXMRPath  [@ReadMem::@mem]
hw.module @ReadMemXMR() {
  hw.instance "ReadMem" sym @ReadMem_sym @ReadMem() -> ()
  // CHECK:      initial
  // CHECK-NEXT:   $readmemb("file3.txt", ReadMem.mem)
  sv.initial {
    %xmr = sv.xmr.ref @ReadMemXMRPath {} : !hw.inout<uarray<8xi32>>
    sv.readmem %xmr, "file3.txt", MemBaseBin : !hw.inout<uarray<8xi32>>
  }
}

hw.hierpath @ReadMem_path [@ReadMemXMRHierPath::@ReadMemXMR_sym, @ReadMemXMR::@ReadMem_sym, @ReadMem::@mem]
// CHECK: module ReadMemXMRHierPath()
hw.module @ReadMemXMRHierPath() {
  hw.instance "ReadMemXMR" sym @ReadMemXMR_sym @ReadMemXMR() -> ()
  // CHECK:      initial
  // CHECK-NEXT:   $readmemb("file4.txt", ReadMemXMRHierPath.ReadMemXMR.ReadMem.mem)
  sv.initial {
    %xmr = sv.xmr.ref @ReadMem_path : !hw.inout<uarray<8xi32>>
    sv.readmem %xmr, "file4.txt", MemBaseBin : !hw.inout<uarray<8xi32>>
  }
}

// CHECK-LABEL: module UninitReg1(
hw.module @UninitReg1(%clock: i1, %reset: i1, %cond: i1, %value: i2) {
  %c-1_i2 = hw.constant -1 : i2
  %count = sv.reg  : !hw.inout<i2>

  // CHECK: always_ff @(posedge clock)
  // CHECK-NEXT:   count <= ~{2{reset}} & (cond ? value : count);

  %0 = sv.read_inout %count : !hw.inout<i2>
  %1 = comb.mux %cond, %value, %0 : i2
  %2 = comb.replicate %reset : (i1) -> i2
  %3 = comb.xor %2, %c-1_i2 : i2
  %4 = comb.and %3, %1 : i2
  sv.alwaysff(posedge %clock)  {
    sv.passign %count, %4 : i2
  }
  hw.output
}

// https://github.com/llvm/circt/issues/2168
// CHECK-LABEL: module shrs_parens(
hw.module @shrs_parens(%a: i18, %b: i18, %c: i1) -> (o: i18) {
  // CHECK: assign o = a + $signed($signed(b) >>> c);
  %c0_i17 = hw.constant 0 : i17
  %0 = comb.concat %c0_i17, %c : i17, i1
  %1 = comb.shrs %b, %0 : i18
  %2 = comb.add %a, %1 : i18
  hw.output %2 : i18
}

// https://github.com/llvm/circt/issues/755
// CHECK-LABEL: module UnaryParensIssue755(
// CHECK: assign b = |(~a);
hw.module @UnaryParensIssue755(%a: i8) -> (b: i1) {
  %c-1_i8 = hw.constant -1 : i8
  %c0_i8 = hw.constant 0 : i8
  %0 = comb.xor %a, %c-1_i8 : i8
  %1 = comb.icmp ne %0, %c0_i8 : i8
  hw.output %1 : i1
}

// Inner name references to ports which are renamed to avoid collisions with
// reserved Verilog keywords.
hw.module.extern @VerbatimModuleExtern(%foo: i1 {hw.exportPort = #hw<innerSym@symA>}) -> (bar: i1 {hw.exportPort = #hw<innerSym@symB>})
// CHECK-LABEL: module VerbatimModule(
// CHECK-NEXT:    input  signed_0
// CHECK-NEXT:    output unsigned_0
hw.module @VerbatimModule(%signed: i1 {hw.exportPort = #hw<innerSym@symA>}) -> (unsigned: i1 {hw.exportPort = #hw<innerSym@symB>}) {
  %parameter = sv.wire sym @symC : !hw.inout<i4>
  %localparam = sv.reg sym @symD : !hw.inout<i4>
  %shortint = sv.interface.instance sym @symE : !sv.interface<@Interface>
  // CHECK: wire [3:0] parameter_0;
  // CHECK: reg  [3:0] localparam_0;
  // CHECK: Interface shortint();
  hw.output %signed : i1
}
sv.verbatim "VERB: module symA `{{0}}`" {symbols = [#hw.innerNameRef<@VerbatimModule::@symA>]}
sv.verbatim "VERB: module symB `{{0}}`" {symbols = [#hw.innerNameRef<@VerbatimModule::@symB>]}
sv.verbatim "VERB: module symC `{{0}}`" {symbols = [#hw.innerNameRef<@VerbatimModule::@symC>]}
sv.verbatim "VERB: module symD `{{0}}`" {symbols = [#hw.innerNameRef<@VerbatimModule::@symD>]}
sv.verbatim "VERB: module symE `{{0}}`" {symbols = [#hw.innerNameRef<@VerbatimModule::@symE>]}
sv.verbatim "VERB: module.extern symA `{{0}}`" {symbols = [#hw.innerNameRef<@VerbatimModuleExtern::@symA>]}
sv.verbatim "VERB: module.extern symB `{{0}}`" {symbols = [#hw.innerNameRef<@VerbatimModuleExtern::@symB>]}
// CHECK: VERB: module symA `signed_0`
// CHECK: VERB: module symB `unsigned_0`
// CHECK: VERB: module symC `parameter_0`
// CHECK: VERB: module symD `localparam_0`
// CHECK: VERB: module symE `shortint_0`
// CHECK: VERB: module.extern symA `foo`
// CHECK: VERB: module.extern symB `bar`


// Should be able to nest interpolated symbols in extra braces
hw.module @CheckNestedBracesSymbol() { hw.output }
sv.verbatim "{{0}} {{{0}}}" {symbols = [@CheckNestedBracesSymbol]}
// CHECK-LABEL: module CheckNestedBracesSymbol();
// CHECK: CheckNestedBracesSymbol {CheckNestedBracesSymbol}


sv.bind #hw.innerNameRef<@BindEmission::@__BindEmissionInstance__> {output_file = #hw.output_file<"BindTest/BindEmissionInstance.sv", excludeFromFileList>}
// CHECK-LABEL: module BindEmissionInstance()
hw.module @BindEmissionInstance() {
  hw.output
}
// CHECK-LABEL: module BindEmission()
hw.module @BindEmission() -> () {
  // CHECK-NEXT: /* This instance is elsewhere emitted as a bind statement
  // CHECK-NEXT:    BindEmissionInstance BindEmissionInstance ();
  // CHECK-NEXT: */
  hw.instance "BindEmissionInstance" sym @__BindEmissionInstance__ @BindEmissionInstance() -> ()  {doNotPrint = true}
  hw.output
}

// Check for instance name matching module name
sv.bind #hw.innerNameRef<@BindEmission2::@BindEmissionInstance> {output_file = #hw.output_file<"BindTest/BindEmissionInstance2.sv", excludeFromFileList>}
// CHECK-LABEL: module BindEmission2()
hw.module @BindEmission2() -> () {
  // CHECK-NEXT: /* This instance is elsewhere emitted as a bind statement
  // CHECK-NEXT:    BindEmissionInstance BindEmissionInstance ();
  // CHECK-NEXT: */
  hw.instance "BindEmissionInstance" sym @BindEmissionInstance @BindEmissionInstance() -> ()  {doNotPrint = true}
  hw.output
}


hw.module @bind_rename_port(%.io_req_ready.output: i1, %reset: i1 { hw.verilogName = "resetSignalName" }, %clock: i1) {
  // CHECK-LABEL: module bind_rename_port
  // CHECK-NEXT: input _io_req_ready_output,
  // CHECK-NEXT:       resetSignalName,
  // CHECK-NEXT:       clock
  hw.output
}

// CHECK-LABEL: module SiFive_MulDiv
hw.module @SiFive_MulDiv(%clock: i1, %reset: i1) -> (io_req_ready: i1) {
  %false = hw.constant false
  hw.instance "InvisibleBind_assert" sym @__ETC_SiFive_MulDiv_assert @bind_rename_port(".io_req_ready.output": %false: i1, reset: %reset: i1, clock: %clock: i1) -> () {doNotPrint = true}
  hw.output %false : i1
  //      CHECK: bind_rename_port InvisibleBind_assert (
  // CHECK-NEXT:   ._io_req_ready_output (1'h0),
  // CHECK-NEXT:   .resetSignalName      (reset),
  // CHECK-NEXT:   .clock                (clock)
  // CHECK-NEXT: );
}

sv.bind.interface <@BindInterface::@__Interface__> {output_file = #hw.output_file<"BindTest/BindInterface.sv", excludeFromFileList>}
sv.interface @Interface {
  sv.interface.signal @a : i1
  sv.interface.signal @b : i1
}

  hw.module.extern @W422_Bar() -> (clock: i1, reset: i1)
  hw.module.extern @W422_Baz() -> (q: i1)
// CHECK-LABEL: module W422_Foo
// CHECK-NOT: GEN
  hw.module @W422_Foo() {
    %false = hw.constant false
    %bar.clock, %bar.reset = hw.instance "bar" @W422_Bar() -> (clock: i1, reset: i1)
    %baz.q = hw.instance "baz" @W422_Baz() -> (q: i1)
    %q = sv.reg sym @__q__  : !hw.inout<i1>
    sv.always posedge %bar.clock, posedge %bar.reset {
      sv.if %bar.reset {
        sv.passign %q, %false : i1
      } else {
        sv.passign %q, %baz.q : i1
      }
    }
    hw.output
  }

hw.module @BindInterface() -> () {
  %bar = sv.interface.instance sym @__Interface__ {doNotPrint = true} : !sv.interface<@Interface>
  hw.output
}

// CHECK-LABEL: FILE "BindTest{{[/\]}}BindEmissionInstance.sv"
// CHECK: bind BindEmission BindEmissionInstance BindEmissionInstance ();

// CHECK-LABEL: FILE "BindTest{{[/\]}}BindInterface.sv"
// CHECK: bind BindInterface Interface bar (.*);

sv.bind #hw.innerNameRef<@SiFive_MulDiv::@__ETC_SiFive_MulDiv_assert>
// CHECK-LABEL: bind SiFive_MulDiv bind_rename_port InvisibleBind_assert
// CHECK-NEXT:  ._io_req_ready_output (1'h0)
// CHECK-NEXT:  .resetSignalName      (reset),
// CHECK-NEXT:  .clock                (clock)
