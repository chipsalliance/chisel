// RUN: circt-opt %s --test-apply-lowering-options='options=emittedLineLength=100,emitBindComments' -export-verilog -verify-diagnostics -o %t.mlir | FileCheck %s

// CHECK-LABEL: // external module E
hw.module.extern @E(%a: i1, %b: i1, %c: i1)
hw.module.extern @Array(%a: !hw.array<2xi4>)

hw.module @TESTSIMPLE(%a: i4, %b: i4, %c: i2, %cond: i1,
                        %array2d: !hw.array<12 x array<10xi4>>,
                        %uarray: !hw.uarray<16xi8>,
                        %postUArray: i8,
                        %structA: !hw.struct<foo: i2, bar:i4>,
                        %arrOfStructA: !hw.array<5 x struct<foo: i2>>,
                        %array1: !hw.array<1xi1>
                        ) -> (
  r0: i4, r2: i4, r4: i4, r6: i4, r7: i4, r8: i4, r9: i4, r10: i4, r11: i4,
  r12: i4, r13: i4, r14: i4, r15: i4, r16: i1, r17: i1, r18: i1, r19: i1,
  r20: i1, r21: i1, r22: i1, r23: i1, r24: i1, r25: i1, r26: i1, r27: i1,
  r28: i1, r29: i12, r30: i2, r31: i9, r33: i4, r34: i4, r35: !hw.array<3xi4>,
  r36: !hw.array<6xi4>, r37: i4, r38: i12, r39: !hw.struct<a: i1, b: i1>,
  r40: !hw.array<4xi2>, r41: !hw.uarray<1xi1>,
  r42: !hw.struct<a: !hw.array<1xi1>>, r43: i4,
  r44: !hw.struct<foo: i2, bar: i4>, r45: !hw.struct<foo: i2, bar: i4>,
  r46: !hw.struct<foo: i2, bar: i4>, r47: i1
  ) attributes {sv.attributes = [#sv.attribute<"svAttr">]} {

  %0 = comb.add %a, %b : i4
  %2 = comb.sub %a, %b : i4
  %4 = comb.mul %a, %b : i4
  %6 = comb.divu %a, %b : i4
  %7 = comb.divs %a, %b : i4
  %8 = comb.modu %a, %b : i4
  %9 = comb.mods %a, %b : i4
  %10 = comb.shl %a, %b : i4
  %11 = comb.shru %a, %b : i4
  %12 = comb.shrs %a, %b : i4
  %13 = comb.or %a, %b : i4
  %14 = comb.and %a, %b : i4
  %15 = comb.xor %a, %b : i4
  %16 = comb.icmp eq %a, %b : i4
  %17 = comb.icmp ne %a, %b : i4
  %18 = comb.icmp slt %a, %b : i4
  %19 = comb.icmp sle %a, %b : i4
  %20 = comb.icmp sgt %a, %b : i4
  %21 = comb.icmp sge %a, %b : i4
  %22 = comb.icmp ult %a, %b : i4
  %23 = comb.icmp ule %a, %b : i4
  %24 = comb.icmp ugt %a, %b : i4
  %25 = comb.icmp uge %a, %b : i4
  %one4 = hw.constant -1 : i4
  %26 = comb.icmp eq %a, %one4 : i4
  %zero4 = hw.constant 0 : i4
  %27 = comb.icmp ne %a, %zero4 : i4
  %28 = comb.parity %a : i4
  %29 = comb.concat %a, %a, %b : i4, i4, i4
  %30 = comb.extract %a from 1 : (i4) -> i2

  %tmp = comb.extract %a from 3 : (i4) -> i1
  %tmp2 = comb.replicate %tmp : (i1) -> i5
  %31 = comb.concat %tmp2, %a : i5, i4
  %33 = comb.mux %cond, %a, %b : i4

  %allone = hw.constant 15 : i4
  %34 = comb.xor %a, %allone : i4

  // Having "sv.namehint" (and checking that it works) anywhere will inherently
  // make tests brittle. This line breaking does not mean your change is no
  // good! You'll just have to find a new place for `sv.namehint`.
  %arrCreated = hw.array_create %allone, %allone, %allone, %allone, %allone, %allone, %allone, %allone, %allone { sv.namehint="name_hint" } : i4
  %slice1 = hw.array_slice %arrCreated[%a] : (!hw.array<9xi4>) -> !hw.array<3xi4>
  %slice2 = hw.array_slice %arrCreated[%b] : (!hw.array<9xi4>) -> !hw.array<3xi4>
  %35 = comb.mux %cond, %slice1, %slice2 {sv.attributes = [#sv.attribute<"svAttr">]} : !hw.array<3xi4>

  %ab = comb.add %a, %b : i4
  %subArr = hw.array_create %allone, %ab, %allone : i4

  %36 = hw.array_concat %subArr, %subArr : !hw.array<3 x i4>, !hw.array<3 x i4>
  %elem2d = hw.array_get %array2d[%a] : !hw.array<12 x array<10xi4>>, i4
  %37 = hw.array_get %elem2d[%b] {sv.attributes = [#sv.attribute<"svAttr">]}: !hw.array<10xi4>, i4

  %38 = comb.replicate %a : (i4) -> i12

  %39 = hw.aggregate_constant [false, true] : !hw.struct<a: i1, b: i1>
  %40 = hw.aggregate_constant [0 : i2, 1 : i2, -2 : i2, -1 : i2] : !hw.array<4xi2>
  %41 = hw.aggregate_constant [false] : !hw.uarray<1xi1>
  %42 = hw.aggregate_constant [[false]] : !hw.struct<a: !hw.array<1xi1>>

  %43 = hw.struct_extract %structA["bar"] : !hw.struct<foo: i2, bar: i4>
  %44 = hw.struct_inject %structA["bar"], %a : !hw.struct<foo: i2, bar: i4>
  %45 = hw.struct_create (%c, %a) : !hw.struct<foo: i2, bar: i4>
  %46 = hw.struct_inject %45["bar"], %b : !hw.struct<foo: i2, bar: i4>
  %none = hw.constant 0 : i0
  %47 = hw.array_get %array1[%none] : !hw.array<1xi1>, i0

  hw.output %0, %2, %4, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, 
            %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30,
            %31, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44,
            %45, %46, %47:
    i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i1, i1, i1, i1, i1, i1,
    i1, i1, i1, i1, i1, i1, i1, i12, i2, i9, i4, i4, !hw.array<3xi4>,
    !hw.array<6xi4>, i4, i12, !hw.struct<a: i1, b: i1>, !hw.array<4xi2>,
    !hw.uarray<1xi1>, !hw.struct<a: !hw.array<1xi1>>, i4, 
    !hw.struct<foo: i2, bar: i4>, !hw.struct<foo: i2, bar: i4>, 
    !hw.struct<foo: i2, bar: i4>, i1
}

// CHECK-LABEL: (* svAttr *)
// CHECK-NEXT: module TESTSIMPLE(
// CHECK-NEXT:      input  [3:0]                                              a,
// CHECK-NEXT:                                                                b,
// CHECK-NEXT:      input  [1:0]                                              c,
// CHECK-NEXT:      input                                                     cond,
// CHECK-NEXT:      input  [11:0][9:0][3:0]                                   array2d,
// CHECK-NEXT:      input  [7:0]                                              uarray[0:15],
// CHECK-NEXT:                                                                postUArray,
// CHECK-NEXT:      input  struct packed {logic [1:0] foo; logic [3:0] bar; } structA,
// CHECK-NEXT:      input  struct packed {logic [1:0] foo; }[4:0]             arrOfStructA,
// CHECK-NEXT:      input  [0:0]                                              array1,
// CHECK-NEXT:      output [3:0]                                              r0,
// CHECK-NEXT:                                                                r2,
// CHECK-NEXT:                                                                r4,
// CHECK-NEXT:                                                                r6,
// CHECK-NEXT:                                                                r7,
// CHECK-NEXT:                                                                r8,
// CHECK-NEXT:                                                                r9,
// CHECK-NEXT:                                                                r10,
// CHECK-NEXT:                                                                r11,
// CHECK-NEXT:                                                                r12,
// CHECK-NEXT:                                                                r13,
// CHECK-NEXT:                                                                r14,
// CHECK-NEXT:                                                                r15,
// CHECK-NEXT:      output                                                    r16,
// CHECK-NEXT:                                                                r17,
// CHECK-NEXT:                                                                r18,
// CHECK-NEXT:                                                                r19,
// CHECK-NEXT:                                                                r20,
// CHECK-NEXT:                                                                r21,
// CHECK-NEXT:                                                                r22,
// CHECK-NEXT:                                                                r23,
// CHECK-NEXT:                                                                r24,
// CHECK-NEXT:                                                                r25,
// CHECK-NEXT:                                                                r26,
// CHECK-NEXT:                                                                r27,
// CHECK-NEXT:                                                                r28,
// CHECK-NEXT:      output [11:0]                                             r29,
// CHECK-NEXT:      output [1:0]                                              r30,
// CHECK-NEXT:      output [8:0]                                              r31,
// CHECK-NEXT:      output [3:0]                                              r33,
// CHECK-NEXT:                                                                r34,
// CHECK-NEXT:      output [2:0][3:0]                                         r35,
// CHECK-NEXT:      output [5:0][3:0]                                         r36,
// CHECK-NEXT:      output [3:0]                                              r37,
// CHECK-NEXT:      output [11:0]                                             r38,
// CHECK-NEXT:      output struct packed {logic a; logic b; }                 r39,
// CHECK-NEXT:      output [3:0][1:0]                                         r40,
// CHECK-NEXT:      output                                                    r41[0:0],
// CHECK-NEXT:      output struct packed {logic [0:0] a; }                    r42,
// CHECK-NEXT:      output [3:0]                                              r43,
// CHECK-NEXT:      output struct packed {logic [1:0] foo; logic [3:0] bar; } r44,
// CHECK-NEXT:                                                                r45,
// CHECK-NEXT:                                                                r46,
// CHECK-NEXT:      output                                                    r47
// CHECK-NEXT: );
// CHECK{LITERAL}:  wire [8:0][3:0] name_hint = {9{4'hF}};
// CHECK-NEXT:      wire [2:0][3:0] [[WIRE0:.+]] = {{[{}][{}]}}4'hF}, {a + b}, {4'hF}};
// CHECK-NEXT:      wire struct packed {logic [1:0] foo; logic [3:0] bar; } [[WIRE1:.+]] = '{foo: c, bar: a};
// CHECK-NEXT:      assign r0 = a + b;
// CHECK-NEXT:      assign r2 = a - b;
// CHECK-NEXT:      assign r4 = a * b;
// CHECK-NEXT:      assign r6 = a / b;
// CHECK-NEXT:      assign r7 = $signed(a) / $signed(b);
// CHECK-NEXT:      assign r8 = a % b;
// CHECK-NEXT:      assign r9 = $signed(a) % $signed(b);
// CHECK-NEXT:      assign r10 = a << b;
// CHECK-NEXT:      assign r11 = a >> b;
// CHECK-NEXT:      assign r12 = $signed($signed(a) >>> b);
// CHECK-NEXT:      assign r13 = a | b;
// CHECK-NEXT:      assign r14 = a & b;
// CHECK-NEXT:      assign r15 = a ^ b;
// CHECK-NEXT:      assign r16 = a == b;
// CHECK-NEXT:      assign r17 = a != b;
// CHECK-NEXT:      assign r18 = $signed(a) < $signed(b);
// CHECK-NEXT:      assign r19 = $signed(a) <= $signed(b);
// CHECK-NEXT:      assign r20 = $signed(a) > $signed(b);
// CHECK-NEXT:      assign r21 = $signed(a) >= $signed(b);
// CHECK-NEXT:      assign r22 = a < b;
// CHECK-NEXT:      assign r23 = a <= b;
// CHECK-NEXT:      assign r24 = a > b;
// CHECK-NEXT:      assign r25 = a >= b;
// CHECK-NEXT:      assign r26 = &a;
// CHECK-NEXT:      assign r27 = |a;
// CHECK-NEXT:      assign r28 = ^a;
// CHECK-NEXT:      assign r29 = {a, a, b};
// CHECK-NEXT:      assign r30 = a[2:1];
// CHECK-NEXT:      assign r31 = {{[{}][{}]}}5{a[3]}}, a};
// CHECK-NEXT:      assign r33 = cond ? a : b;
// CHECK-NEXT:      assign r34 = ~a;
// CHECK-NEXT:      assign r35 = cond ? (* svAttr *) name_hint[a +: 3] : name_hint[b +: 3];
// CHECK-NEXT:      assign r36 = {[[WIRE0]], [[WIRE0]]};
// CHECK-NEXT:      assign r37 = array2d[a][b] (* svAttr *);
// CHECK-NEXT:      assign r38 = {3{a}};
// CHECK-NEXT:      assign r39 = '{a: 1'h0, b: 1'h1};
// CHECK-NEXT:      assign r40 = {2'h0, 2'h1, 2'h2, 2'h3};
// CHECK-NEXT:      assign r41 = '{1'h0};
// CHECK-NEXT:      assign r42 = '{a: {1'h0}};
// CHECK-NEXT:      assign r43 = structA.bar;
// CHECK-NEXT:      assign r44 = '{foo: structA.foo, bar: a};
// CHECK-NEXT:      assign r45 = _GEN_0;
// CHECK-NEXT:      assign r46 = '{foo: [[WIRE1]].foo, bar: b};
// CHECK-NEXT:      assign r47 = array1[/*Zero width*/ 1'b0];
// CHECK-NEXT: endmodule

hw.module @i0Inst() {
  %c0_i0 = hw.constant 0 : i0
  hw.instance "i0" sym @i0Module @i0Module(arg1: %c0_i0: i0) -> ()
}
hw.module @i0Module(%arg1: i0) -> () {}
// CHECK-LABEL: module i0Inst();
// CHECK-NEXT:    i0Module i0 (
// CHECK-NEXT:    //.arg1 (/* Zero width */)
// CHECK-NEXT:     );
// CHECK-NEXT:  endmodule
// CHECK-LABEL: module i0Module(
// CHECK-NEXT:    // input /*Zero Width*/ arg1
// CHECK-NEXT:     );
// CHECK-EMPTY:
// CHECK-NEXT:  endmodule

hw.module @B(%a: i1) -> (b: i1, c: i1) {
  %0 = comb.or %a, %a : i1
  %1 = comb.and %a, %a : i1
  hw.output %0, %1 : i1, i1
}
// CHECK-LABEL: module B(
// CHECK-NEXT:   input  a, //
// CHECK-NEXT:   output b, //
// CHECK-NEXT:          c //
// CHECK-NEXT:  );
// CHECK-EMPTY:
// CHECK-NEXT:   assign b = a | a;
// CHECK-NEXT:   assign c = a & a;
// CHECK-NEXT: endmodule

hw.module @A(%d: i1, %e: i1) -> (f: i1) {
  %1 = comb.mux %d, %d, %e : i1
  hw.output %1 : i1
}
// CHECK-LABEL: module A(
// CHECK-NEXT:  input  d,
// CHECK-NEXT:         e,
// CHECK-NEXT:  output f
// CHECK-NEXT:  );
// CHECK-EMPTY:
// CHECK-NEXT:  assign f = d ? d : e;
// CHECK-NEXT: endmodule

hw.module @AAA(%d: i1, %e: i1) -> (f: i1) {
  %z = hw.constant 0 : i1
  hw.output %z : i1
}
// CHECK-LABEL: module AAA(
// CHECK-NEXT:  input  d,
// CHECK-NEXT:         e,
// CHECK-NEXT:  output f
// CHECK-NEXT:  );
// CHECK-EMPTY:
// CHECK-NEXT:  assign f = 1'h0;
// CHECK-NEXT: endmodule


/// TODO: Specify parameter declarations.
hw.module.extern @EXT_W_PARAMS<DEFAULT: i64, DEPTH: f64, FORMAT: none,
     WIDTH: i8>(%a: i1, %b: i0) -> (out: i1)
  attributes { verilogName="FooModule" }

hw.module.extern @EXT_W_PARAMS2<DEFAULT: i32>(%a: i2) -> (out: i1)
  attributes { verilogName="FooModule" }

hw.module @AB(%w: i1, %x: i1, %i2: i2, %i3: i0) -> (y: i1, z: i1, p: i1, p2: i1) {
  %w2 = hw.instance "a1" @AAA(d: %w: i1, e: %w1: i1) -> (f: i1)
  %w1, %y = hw.instance "b1" @B(a: %w2: i1) -> (b: i1, c: i1)

  %p = hw.instance "paramd" @EXT_W_PARAMS<
   DEFAULT: i64 = 14000240888948784983, DEPTH: f64 = 3.242000e+01,
   FORMAT: none = "xyz_timeout=%d\0A", WIDTH: i8 = 32
  >(a: %w: i1, b: %i3: i0) -> (out: i1)

  %p2 = hw.instance "paramd2" @EXT_W_PARAMS2<DEFAULT: i32 = 1>(a: %i2: i2) -> (out: i1)

  hw.output %y, %x, %p, %p2 : i1, i1, i1, i1
}
// CHECK-LABEL: module AB(
// CHECK-NEXT:      input                 w,
// CHECK-NEXT:                            x,
// CHECK-NEXT:      input  [1:0]          i2,
// CHECK-NEXT:   // input  /*Zero Width*/ i3,
// CHECK-NEXT:      output                y,
// CHECK-NEXT:                            z,
// CHECK-NEXT:                            p,
// CHECK-NEXT:                            p2
// CHECK-NEXT:   );
// CHECK-EMPTY:
// CHECK-NEXT:   wire _b1_b;
// CHECK-NEXT:   wire _a1_f;
// CHECK:        AAA a1 (
// CHECK-NEXT:     .d (w),
// CHECK-NEXT:     .e (_b1_b),
// CHECK-NEXT:     .f (_a1_f)
// CHECK-NEXT:   );
// CHECK-NEXT:   B b1 (
// CHECK-NEXT:     .a (_a1_f),
// CHECK-NEXT:     .b (_b1_b),
// CHECK-NEXT:     .c (y)
// CHECK-NEXT:   );
// CHECK-NEXT:   FooModule #(
// CHECK-NEXT:     .DEFAULT(-64'd4446503184760766633),
// CHECK-NEXT:     .DEPTH(3.242000e+01),
// CHECK-NEXT:     .FORMAT("xyz_timeout=%d\n"),
// CHECK-NEXT:     .WIDTH(32)
// CHECK-NEXT:   ) paramd (
// CHECK-NEXT:     .a   (w),
// CHECK-NEXT:   //.b   (i3),
// CHECK-NEXT:     .out (p)
// CHECK-NEXT:   );
// CHECK-NEXT:   FooModule #(
// CHECK-NEXT:     .DEFAULT(1)
// CHECK-NEXT:   ) paramd2 (
// CHECK-NEXT:     .a   (i2),
// CHECK-NEXT:     .out (p2)
// CHECK-NEXT:   );
// CHECK-NEXT:   assign z = x;
// CHECK-NEXT: endmodule



hw.module @shl(%a: i1) -> (b: i1) {
  %0 = comb.shl %a, %a : i1
  hw.output %0 : i1
}
// CHECK-LABEL:  module shl(
// CHECK-NEXT:   input  a,
// CHECK-NEXT:   output b
// CHECK-NEXT:   );
// CHECK-EMPTY:
// CHECK-NEXT:   assign b = a << a;
// CHECK-NEXT: endmodule


hw.module @inout_0(%a: !hw.inout<i42>) -> (out: i42) {
  %aget = sv.read_inout %a: !hw.inout<i42>
  hw.output %aget : i42
}
// CHECK-LABEL:  module inout_0(
// CHECK-NEXT:     inout  [41:0] a,
// CHECK-NEXT:     output [41:0] out
// CHECK-NEXT:   );
// CHECK-EMPTY:
// CHECK-NEXT:     assign out = a;
// CHECK-NEXT:   endmodule

// https://github.com/llvm/circt/issues/316
// FIXME: The MLIR parser doesn't accept an i0 even though it is valid IR,
// this needs to be fixed upstream.
//hw.module @issue316(%inp_0: i0) {
//  hw.output
//}

// https://github.com/llvm/circt/issues/318
// This shouldn't generate invalid Verilog
hw.module @extract_all(%tmp85: i1) -> (tmp106: i1) {
  %1 = comb.extract %tmp85 from 0 : (i1) -> i1
  hw.output %1 : i1
}
// CHECK-LABEL: module extract_all
// CHECK:  assign tmp106 = tmp85;

hw.module @wires(%in4: i4, %in8: i8) -> (a: i4, b: i8, c: i8) {
  // CHECK-LABEL: module wires( //
  // CHECK-NEXT:   input  [3:0] in4, //
  // CHECK-NEXT:   input  [7:0] in8, //
  // CHECK-NEXT:   output [3:0] a, //
  // CHECK-NEXT:   output [7:0] b, //
  // CHECK-NEXT:                c //
  // CHECK-NEXT:  );

  // CHECK-EMPTY:

  // Wires.
  // CHECK-NEXT: wire [3:0]            myWire = in4;
  %myWire = sv.wire : !hw.inout<i4>

  // Packed arrays.

  // CHECK-NEXT: wire [41:0][7:0]      myArray1;
  %myArray1 = sv.wire : !hw.inout<array<42 x i8>>
  // CHECK-NEXT: wire [2:0][41:0][3:0] myWireArray2;
  %myWireArray2 = sv.wire : !hw.inout<array<3 x array<42 x i4>>>

  // Unpacked arrays, and unpacked arrays of packed arrays.

  // CHECK-NEXT: wire [7:0]            myUArray1[0:41];
  %myUArray1 = sv.wire : !hw.inout<uarray<42 x i8>>

  // CHECK-NEXT: wire [9:0][7:0]       myUArray2[0:13][0:11];
  %myUArray2 = sv.wire : !hw.inout<uarray<14 x uarray<12 x array<10 x i8>>>>

  // Wires.
  sv.assign %myWire, %in4 : i4
  %wireout = sv.read_inout %myWire : !hw.inout<i4>

  // Packed arrays.

  %subscript = sv.array_index_inout %myArray1[%in4] : !hw.inout<array<42 x i8>>, i4
  // CHECK-NEXT: assign myArray1[in4] = in8;
  sv.assign %subscript, %in8 : i8

  %memout1 = sv.read_inout %subscript : !hw.inout<i8>

    // Unpacked arrays, and unpacked arrays of packed arrays.
  %subscriptu = sv.array_index_inout %myUArray1[%in4] : !hw.inout<uarray<42 x i8>>, i4
  // CHECK-NEXT: assign myUArray1[in4] = in8;
  sv.assign %subscriptu, %in8 : i8

  %memout2 = sv.read_inout %subscriptu : !hw.inout<i8>

  // CHECK-NEXT: assign a = myWire;
  // CHECK-NEXT: assign b = myArray1[in4];
  // CHECK-NEXT: assign c = myUArray1[in4];
  hw.output %wireout, %memout1, %memout2 : i4, i8, i8
}

// CHECK-LABEL: module signs
hw.module @signs(%in1: i4, %in2: i4, %in3: i4, %in4: i4)  {
  %awire = sv.wire : !hw.inout<i4>
  // CHECK: wire [3:0] awire;

  // CHECK: assign awire = $unsigned($signed(in1) / $signed(in2)) /
  // CHECK:                $unsigned($signed(in3) / $signed(in4));
  %a1 = comb.divs %in1, %in2: i4
  %a2 = comb.divs %in3, %in4: i4
  %a3 = comb.divu %a1, %a2: i4
  sv.assign %awire, %a3: i4

  // CHECK:       assign awire =
  // CHECK-NEXT:    $unsigned($signed(in1) / $signed(in2) + $signed(in1) / $signed(in2))
  // CHECK-NEXT:    / $unsigned($signed(in1) / $signed(in2) * $signed(in1) / $signed(in2));
  %b1a = comb.divs %in1, %in2: i4
  %b1b = comb.divs %in1, %in2: i4
  %b1c = comb.divs %in1, %in2: i4
  %b1d = comb.divs %in1, %in2: i4
  %b2 = comb.add %b1a, %b1b: i4
  %b3 = comb.mul %b1c, %b1d: i4
  %b4 = comb.divu %b2, %b3: i4
  sv.assign %awire, %b4: i4

  // https://github.com/llvm/circt/issues/369
  // CHECK: assign awire = 4'sh5 / -4'sh3;
  %c5_i4 = hw.constant 5 : i4
  %c-3_i4 = hw.constant -3 : i4
  %divs = comb.divs %c5_i4, %c-3_i4 : i4
  sv.assign %awire, %divs: i4

  hw.output
}


// CHECK-LABEL: module casts(
// CHECK-NEXT: input  [6:0]      in1,
// CHECK-NEXT: input  [7:0][3:0] in2,
// CHECK-NEXT: output [6:0]      r1,
// CHECK-NEXT: output [31:0]     r2
// CHECK-NEXT: );
hw.module @casts(%in1: i7, %in2: !hw.array<8xi4>) -> (r1: !hw.array<7xi1>, r2: i32) {
  // CHECK-EMPTY:
  %r1 = hw.bitcast %in1 : (i7) -> !hw.array<7xi1>
  %r2 = hw.bitcast %in2 : (!hw.array<8xi4>) -> i32

  // CHECK-NEXT: assign r1 = in1;
  // CHECK-NEXT: assign r2 = /*cast(bit[31:0])*/in2;
  hw.output %r1, %r2 : !hw.array<7xi1>, i32
}

// CHECK-LABEL: module TestZero(
// CHECK-NEXT:      input  [3:0]               a,
// CHECK-NEXT:   // input  /*Zero Width*/      zeroBit,
// CHECK-NEXT:   // input  [2:0]/*Zero Width*/ arrZero,
// CHECK-NEXT:      output [3:0]               r0
// CHECK-NEXT:   // output /*Zero Width*/      rZero
// CHECK-NEXT:   // output [2:0]/*Zero Width*/ arrZero_0
// CHECK-NEXT:   );
// CHECK-EMPTY:
hw.module @TestZero(%a: i4, %zeroBit: i0, %arrZero: !hw.array<3xi0>)
  -> (r0: i4, rZero: i0, arrZero_0: !hw.array<3xi0>) {

  %b = comb.add %a, %a : i4
  hw.output %b, %zeroBit, %arrZero : i4, i0, !hw.array<3xi0>

  // CHECK-NEXT:   assign r0 = a + a;
  // CHECK-NEXT:   // Zero width: assign rZero = zeroBit;
  // CHECK-NEXT:   // Zero width: assign arrZero_0 = arrZero;
  // CHECK-NEXT: endmodule
}

// CHECK-LABEL: TestZeroInstance
hw.module @TestZeroInstance(%aa: i4, %azeroBit: i0, %aarrZero: !hw.array<3xi0>)
  -> (r0: i4, rZero: i0, arrZero_0: !hw.array<3xi0>) {

// CHECK:  TestZero iii (
// CHECK-NEXT:    .a         (aa),
// CHECK-NEXT:  //.zeroBit   (azeroBit),
// CHECK-NEXT:  //.arrZero   (aarrZero),
// CHECK-NEXT:    .r0        (r0)
// CHECK-NEXT:  //.rZero     (rZero)
// CHECK-NEXT:  //.arrZero_0 (arrZero_0)
// CHECK-NEXT:  );
// CHECK-NEXT: endmodule

  %o1, %o2, %o3 = hw.instance "iii" @TestZero(a: %aa: i4, zeroBit: %azeroBit: i0, arrZero: %aarrZero: !hw.array<3xi0>) -> (r0: i4, rZero: i0, arrZero_0: !hw.array<3xi0>)

  hw.output %o1, %o2, %o3 : i4, i0, !hw.array<3xi0>
}

// CHECK: module TestZeroStruct(
// CHECK-NEXT:  // input  /*Zero Width*/ structZero,
// CHECK-NEXT:  // input  /*Zero Width*/ structZeroNest,
// CHECK-NEXT:  // output /*Zero Width*/ structZero_0,
// CHECK-NEXT:  // output /*Zero Width*/ structZeroNest_0
// CHECK-NEXT: );
hw.module @TestZeroStruct(%structZero: !hw.struct<>, %structZeroNest: !hw.struct<a: !hw.struct<>>)
  -> (structZero_0: !hw.struct<>, structZeroNest_0: !hw.struct<a: !hw.struct<>>) {

  hw.output %structZero, %structZeroNest : !hw.struct<>, !hw.struct<a: !hw.struct<>>
  // CHECK:      // Zero width: assign structZero_0 = structZero;
  // CHECK-NEXT: // Zero width: assign structZeroNest_0 = structZeroNest;
  // CHECK-NEXT: endmodule
}

// CHECK-LABEL: module zeroElements
// CHECK-NEXT:   // input  /*Zero Width*/                                                                                              in0,
// CHECK-NEXT:      input  [31:0]                                                                                                      in1,
// CHECK-NEXT:      output struct packed {/*z1: Zero Width;*/ logic [31:0] a; /*z2: Zero Width;*/ logic [31:0] b; /*c: Zero Width;*/ struct packed {logic [31:0] d1; /*z: Zero Width;*/ } d; } out0
// CHECK-NEXT:  );
hw.module @zeroElements(%in0: i0, %in1: i32) -> (out0: !hw.struct<z1: i0, a: i32, z2: i0, b: i32, c: !hw.struct<z: i0>, d: !hw.struct<d1:i32, z:i0>>) {
  // CHECK:      // Zero width: wire /*Zero Width*/
  // CHECK-SAME: _GEN = '{};
  // CHECK-NEXT: wire struct packed {logic [31:0] d1; /*z: Zero Width;*/ }
  // CHECK-NEXT:   _GEN_0 = '{d1: in1};
  //      CHECK: wire
  // CHECK-NEXT:   struct packed {/*z1: Zero Width;*/ logic [31:0] a; /*z2: Zero Width;*/ logic [31:0] b; /*c: Zero Width;*/ struct packed {logic [31:0] d1; /*z: Zero Width;*/ } d; } 
  // CHECK-NEXT:   _GEN_1 = '{a: in1, b: in1, d: _GEN_0};
  // CHECK-NEXT: assign out0 = '{a: in1, b: _GEN_1.b, d: _GEN_1.d};
  %0 = hw.struct_create (%in0) : !hw.struct<z: i0>
  %1 = hw.struct_create (%in1, %in0) : !hw.struct<d1:i32, z:i0>
  %2 = hw.struct_create (%in0, %in1, %in0, %in1, %0, %1) : !hw.struct<z1: i0, a: i32, z2: i0, b: i32, c: !hw.struct<z: i0>, d: !hw.struct<d1:i32, z:i0>>
  %3 = hw.struct_inject %2["a"], %in1 : !hw.struct<z1: i0, a: i32, z2: i0, b: i32, c: !hw.struct<z: i0>, d: !hw.struct<d1:i32, z:i0>>
  hw.output %3 : !hw.struct<z1: i0, a: i32, z2: i0, b: i32, c: !hw.struct<z: i0>, d: !hw.struct<d1:i32, z:i0>>
}

// CHECK-LABEL: TestZeroStructInstance
hw.module @TestZeroStructInstance(%structZero: !hw.struct<>, %structZeroNest: !hw.struct<a: !hw.struct<>>)
  -> (structZero_0: !hw.struct<>, structZeroNest_0: !hw.struct<a: !hw.struct<>>) {

// CHECK: TestZeroStruct iii (
// CHECK-NEXT:  //.structZero       (structZero)
// CHECK-NEXT:  //.structZeroNest   (structZeroNest)
// CHECK-NEXT:  //.structZero_0     (structZero_0)
// CHECK-NEXT:  //.structZeroNest_0 (structZeroNest_0)
// CHECK-NEXT:  );

  %o1, %o2 = hw.instance "iii" @TestZeroStruct(structZero: %structZero: !hw.struct<>, structZeroNest: %structZeroNest: !hw.struct<a: !hw.struct<>>)
                                -> (structZero_0: !hw.struct<>, structZeroNest_0: !hw.struct<a: !hw.struct<>>)

  hw.output %o1, %o2 : !hw.struct<>, !hw.struct<a: !hw.struct<>>
}

// CHECK-LABEL: module testZeroArrayGet(
// CHECK-NEXT:   // input  /*Zero Width*/ arg0,
// CHECK-NEXT:      input  [0:0][31:0]    arg1,
// CHECK-NEXT:      output [31:0]         out,
// CHECK-NEXT:                            out1
// CHECK-NEXT:   // output /*Zero Width*/ out2
// CHECK-NEXT:  );

// CHECK:   assign out = arg1[/*Zero width*/ 1'b0];	
// CHECK-NEXT:   assign out1 = arg1[/*Zero width*/ 1'b0];	
// CHECK-NEXT:   // Zero width: assign out2 = arg0;	

hw.module @testZeroArrayGet(%arg0: i0, %arg1 : !hw.array<1xi32>) -> (out: i32, out1: i32, out2: i0) {
  // Using an expression as index.
  %idx = comb.add %arg0, %arg0 : i0
  %0 = hw.array_get %arg1[%idx] : !hw.array<1xi32>, i0

  // Using an argument as index.
  %1 = hw.array_get %arg1[%arg0] : !hw.array<1xi32>, i0
  hw.output %0, %1, %arg0 : i32, i32, i0
}

// https://github.com/llvm/circt/issues/438
// CHECK-LABEL: module cyclic
hw.module @cyclic(%a: i1) -> (b: i1) {
  // CHECK: wire _GEN;

  // CHECK: wire _GEN_0 = _GEN + _GEN;
  %1 = comb.add %0, %0 : i1
  // CHECK: assign _GEN = a << a;
  %0 = comb.shl %a, %a : i1
  // CHECK: assign b = _GEN_0 - _GEN_0;
  %2 = comb.sub %1, %1 : i1
  hw.output %2 : i1
}


// https://github.com/llvm/circt/issues/668
// CHECK-LABEL: module longExpressions
hw.module @longExpressions(%a: i8, %a2: i8) -> (b: i8) {
  // CHECK:      assign b =
  // CHECK-NEXT:   (a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a)
  // CHECK-NEXT:   * (a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a)
  // CHECK-NEXT:   | (a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a)
  // CHECK-NEXT:   * (a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a);

  %1 = comb.add %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a : i8
  %2 = comb.add %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a : i8
  %3 = comb.add %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a : i8
  %4 = comb.add %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a : i8
  %5 = comb.mul %1, %2 : i8
  %6 = comb.mul %3, %4 : i8
  %7 = comb.or %5, %6 : i8
  hw.output %7 : i8
}

// https://github.com/llvm/circt/issues/668
// CHECK-LABEL: module longvariadic
hw.module @longvariadic(%a: i8) -> (b: i8) {
  // CHECK:          assign b =
  // CHECK-NEXT:       a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a
  // CHECK-COUNT-9:    + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a
  // CHECK-NEXT:       + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a;

  %1 = comb.add %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a,
                %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a,
                %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a,
                %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a,
                %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a,
                %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a,
                %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a,
                %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a,
                %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a,
                %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a,
                %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a,
                %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a,
                %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a,
                %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a,
                %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a,
                %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a : i8
  hw.output %1 : i8
}

// https://github.com/llvm/circt/issues/736
// Can't depend on left associativeness since ops can have args with different sizes
// CHECK-LABEL: module eqIssue(
// CHECK-NEXT: input  [8:0] a,
// CHECK-NEXT:              c,
// CHECK-NEXT: input  [3:0] d,
// CHECK-NEXT:              e,
// CHECK-NEXT: output       r
// CHECK-NEXT: );
// CHECK-EMPTY:
// CHECK-NEXT: assign r = a == c == (d == e);
  hw.module @eqIssue(%a: i9, %c :i9, %d: i4, %e: i4) -> (r : i1){
    %1 = comb.icmp eq %a, %c : i9
    %2 = comb.icmp eq %d, %e : i4
    %4 = comb.icmp eq %1, %2 : i1
    hw.output %4 : i1
  }

// https://github.com/llvm/circt/issues/750
// Always get array indexes on the lhs
// CHECK-LABEL: module ArrayLHS
// CHECK-NEXT:    input clock
// CHECK-NEXT:  );
// CHECK-EMPTY:
// CHECK-NEXT:   reg memory_r_en_pipe[0:0];
// CHECK:        always_ff @(posedge clock)
// CHECK-NEXT:     memory_r_en_pipe[1'h0] <= 1'h0;
// CHECK-NEXT:   initial
// CHECK-NEXT:     memory_r_en_pipe[1'h0] = 1'h0;
// CHECK-NEXT: endmodule
hw.module @ArrayLHS(%clock: i1) {
  %false = hw.constant false
  %memory_r_en_pipe = sv.reg  : !hw.inout<uarray<1xi1>>
  %3 = sv.array_index_inout %memory_r_en_pipe[%false] : !hw.inout<uarray<1xi1>>, i1
  sv.alwaysff(posedge %clock)  {
    sv.passign %3, %false : i1
  }
  sv.initial  {
    sv.bpassign %3, %false : i1
  }
}

// CHECK-LABEL: module noTemporaryIfReadInOutIsAfterUse
hw.module @noTemporaryIfReadInOutIsAfterUse(%clock: i1, %x: i1) {
  // CHECK: wire aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa;

  %0 = comb.and %1, %x : i1
  // CHECK: always_ff @(posedge clock) begin
  sv.alwaysff(posedge %clock) {
    // CHECK: if (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa & x) begin
    sv.if %0  {
      sv.verbatim "// hello"
    }
  }

  // CHECK: end // always_ff @(posedge)
  %aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa = sv.wire  : !hw.inout<i1>
  %1 = sv.read_inout %aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa : !hw.inout<i1>
}

// CHECK-LABEL: module largeConstant
hw.module @largeConstant(%a: i100000, %b: i16) -> (x: i100000, y: i16) {
  // Large constant is inlined on multiple lines.

  // CHECK:      assign x =
  // CHECK-NEXT:   a + 100000'h2CD76FE086B93CE2F768A00B22A00000000000
  // CHECK-NEXT:   + 100000'h2CD76FE086B93CE2F768A00B22A00000000000
  // CHECK-NEXT:   + 100000'h2CD76FE086B93CE2F768A00B22A00000000000
  // CHECK-NEXT:   + 100000'h2CD76FE086B93CE2F768A00B22A00000000000
  // CHECK-NEXT:   + 100000'h2CD76FE086B93CE2F768A00B22A00000000000
  // CHECK-NEXT:   + 100000'h2CD76FE086B93CE2F768A00B22A00000000000
  // CHECK-NEXT:   + 100000'h2CD76FE086B93CE2F768A00B22A00000000000
  // CHECK-NEXT:   + 100000'h2CD76FE086B93CE2F768A00B22A00000000000;
  %c = hw.constant 1000000000000000000000000000000000000000000000 : i100000
  %1 = comb.add %a, %c, %c, %c, %c, %c, %c, %c, %c : i100000

  // Small constants are emitted inline.

  // CHECK: assign y = b + 16'hA + 16'hA + 16'hA + 16'hA + 16'hA + 16'hA + 16'hA + 16'hA;
  %c2 = hw.constant 10 : i16
  %2 = comb.add %b, %c2, %c2, %c2, %c2, %c2, %c2, %c2, %c2 : i16

  hw.output %1, %2 : i100000, i16
}
// CHECK-LABEL: StructExtractExtract
hw.module @StructExtractExtract(%a: !hw.struct<b: i4>) -> (r: i2) {
  %0 = hw.struct_extract %a["b"] : !hw.struct<b: i4>
  %1 = comb.extract %0 from 1 : (i4) -> i2
  // CHECK: assign r = a.b[2:1];
  hw.output %1 : i2
}

// CHECK-LABEL: StrurctExtractInline
hw.module @StrurctExtractInline(%a: !hw.struct<v: i1>) -> (b: i1, c: i1) {
  %0 = hw.struct_extract %a["v"] : !hw.struct<v: i1>
  // CHECK:      assign b = a.v;
  // CHECK-NEXT: assign c = a.v;
  hw.output %0, %0 : i1, i1
}

// CHECK-LABEL: NoExtraTemporaryWireForAssign
hw.module @NoExtraTemporaryWireForAssign(%a: i2, %b: i4) {
  // CHECK: wire struct packed {logic [1:0] foo; logic [3:0] bar; } _GEN = '{foo: a, bar: b};
  %0 = hw.struct_create (%a, %b) : !hw.struct<foo: i2, bar: i4>
  %1 = sv.wire : !hw.inout<!hw.struct<foo: i2, bar: i4>>
  sv.assign %1, %0: !hw.struct<foo: i2, bar: i4>
}

hw.module.extern @DifferentResultMod() -> (out1: i1, out2: i2)

// CHECK-LABEL: module out_of_order_multi_result(
hw.module @out_of_order_multi_result() -> (b: i1, c: i2) {
  // CHECK: wire       _b1_out1;
  // CHECK: wire [1:0] _b1_out2;
  %b = comb.add %out1, %out1 : i1
  %c = comb.add %out2, %out2 : i2

  %out1, %out2 = hw.instance "b1" @DifferentResultMod() -> (out1: i1, out2: i2)

  // CHECK: assign b = _b1_out1 + _b1_out1;
  // CHECK: assign c = _b1_out2 + _b1_out2;
  hw.output %b, %c : i1, i2
}

hw.module.extern @single_result() -> (res: i3)
// CHECK-LABEL: module instance_result_reuse_wires(
hw.module @instance_result_reuse_wires() -> (b: i3) {
  // CHECK:       wire {{.*}} some_wire;
  // CHECK:       single_result b1 (
  // CHECK-NEXT:  .res (some_wire)
  // CHECK-NEXT:  );
  // CHECK-NEXT:  assign b = some_wire;
  %some_wire = sv.wire : !hw.inout<i3>
  %read = sv.read_inout %some_wire : !hw.inout<i3>

  %out1 = hw.instance "b1" @single_result() -> (res: i3)
  sv.assign %some_wire, %out1 : i3

  hw.output %read : i3
}

hw.module.extern @ExternDestMod(%a: i1, %b: i2) -> (c: i3, d: i4)
hw.module @InternalDestMod(%a: i1, %b: i3, %c: i1) {}
// CHECK-LABEL: module ABC
hw.module @ABC(%a: i1, %b: i2) -> (c: i4) {
  %0,%1 = hw.instance "whatever" sym @a1 @ExternDestMod(a: %a: i1, b: %b: i2) -> (c: i3, d: i4) {doNotPrint=1}
  %2 = sv.xmr "whatever", "a" : !hw.inout<i1>
  %3 = sv.read_inout %2: !hw.inout<i1>
  hw.instance "yo" sym @b1 @InternalDestMod(a: %a: i1, b: %0: i3, c: %3: i1) -> () {doNotPrint=1}
  hw.output %1 : i4
}

// CHECK:   wire [2:0] _whatever_c;
// CHECK:        /* This instance is elsewhere emitted as a bind statement
// CHECK-NEXT:      ExternDestMod whatever (
// CHECK-NEXT:        .a (a),
// CHECK-NEXT:        .b (b),
// CHECK-NEXT:        .c (_whatever_c),
// CHECK-NEXT:        .d (c)
// CHECK-NEXT:      );
// CHECK-NEXT:   */
// CHECK-NEXT:   /* This instance is elsewhere emitted as a bind statement
// CHECK-NEXT:      InternalDestMod yo (
// CHECK-NEXT:        .a (a),
// CHECK-NEXT:        .b (_whatever_c)
// CHECK-NEXT:        .c (whatever.a)
// CHECK-NEXT:      );
// CHECK-NEXT:   */
// CHECK-NEXT: endmodule


hw.module.extern @Uwu() -> (uwu_output : i32)
hw.module.extern @Owo(%owo_in : i32) -> ()

// CHECK-LABEL: module Nya(
hw.module @Nya() -> (nya_output : i32) {
  %0 = hw.instance "uwu" @Uwu() -> (uwu_output: i32)
  // CHECK: wire [31:0] _uwu_uwu_output;
  // CHECK: Uwu uwu (
  // CHECK: .uwu_output (_uwu_uwu_output)
  // CHECK: );

  hw.instance "owo" @Owo(owo_in: %0: i32) -> ()
  // CHECK: Owo owo (
  // CHECK: .owo_in (_uwu_uwu_output)
  // CHECK: );

  hw.output %0 : i32
  // CHECK: assign nya_output = _uwu_uwu_output;
  // CHECK: endmodule
}

// CHECK-LABEL: module Nya2(
hw.module @Nya2() -> (nya2_output : i32) {
  %0 = hw.instance "uwu" @Uwu() -> (uwu_output: i32)
  // CHECK: Uwu uwu (
  // CHECK: .uwu_output (nya2_output)
  // CHECK: );

  hw.output %0 : i32
  // CHECK: endmodule
}

hw.module.extern @Ni() -> (ni_output : i0)
hw.module.extern @San(%san_input : i0) -> ()

// CHECK-LABEL: module Ichi(
hw.module @Ichi() -> (Ichi_output : i0) {
  %0 = hw.instance "ni" @Ni() -> (ni_output: i0)
  // CHECK: Ni ni (
  // CHECK: //.ni_output (Ichi_output)
  // CHECK-NEXT: );

  hw.output %0 : i0
  // CHECK: endmodule
}

// CHECK-LABEL: module Chi(
hw.module @Chi() -> (Chi_output : i0) {
  %0 = hw.instance "ni" @Ni() -> (ni_output: i0)
  // CHECK: Ni ni (
  // CHECK: //.ni_output (_ni_ni_output)
  // CHECK-NEXT: );

  hw.instance "san" @San(san_input: %0: i0) -> ()
  // CHECK: San san (
  // CHECK: //.san_input (_ni_ni_output)
  // CHECK-NEXT: );

  // CHECK: // Zero width: assign Chi_output = _ni_ni_output;
  hw.output %0 : i0
  // CHECK: endmodule
}

// CHECK-LABEL: module Choochoo(
hw.module @Choochoo() -> (out : i0) {
  %0 = hw.constant 0 : i0
  // CHECK: // Zero width: assign out = /*Zero width*/;
  hw.output %0 : i0
}

// CHECK-LABEL: module Foo1360(
// Issue #1360: https://github.com/llvm/circt/issues/1360

 hw.module @Foo1360() {
   // CHECK:      RealBar #(
   // CHECK-NEXT:   .WIDTH0(0),
   // CHECK-NEXT:   .WIDTH1(4),
   // CHECK-NEXT:   .WIDTH2(40'd6812312123),
   // CHECK-NEXT:   .WIDTH3(-1),
   // CHECK-NEXT:   .WIDTH4(-68'sd88888888888888888),
   // CHECK-NEXT:   .Wtricky(40'd4294967295)
   // CHECK-NEXT: ) bar ();

   hw.instance "bar" @Bar1360<
     WIDTH0: i32 = 0, WIDTH1: i4 = 4, WIDTH2: i40 = 6812312123, WIDTH3: si4 = -1,
     WIDTH4: si68 = -88888888888888888, Wtricky: i40 = 4294967295
   >() -> ()
   hw.output
 }
 hw.module.extern @Bar1360<
     WIDTH0: i32, WIDTH1: i4, WIDTH2: i40, WIDTH3: si4, WIDTH4: si68, Wtricky: i40
   >() attributes {verilogName = "RealBar"}

// CHECK-LABEL: module Issue1563(
hw.module @Issue1563(%a: i32) -> (out : i32) {
  // CHECK: assign out = a + a;{{.*}}//{{.*}}XX.scala:123:19, YY.haskell:309:14, ZZ.swift:3:4
  %0 = comb.add %a, %a : i32 loc(fused["XX.scala":123:19, "YY.haskell":309:14, "ZZ.swift":3:4])
  hw.output %0 : i32
  // CHECK: endmodule
}

// CHECK-LABEL: module Foo1587
// Issue #1587: https://github.com/llvm/circt/issues/1587
hw.module @Foo1587(%idx: i2, %a_0: i4, %a_1: i4, %a_2: i4, %a_3: i4) -> (b: i4) {
  %0 = hw.array_create %a_0, %a_1, %a_2, %a_3 : i4
  %1 = hw.array_get %0[%idx] : !hw.array<4xi4>, i2
  hw.output %1 : i4
  // CHECK: wire [3:0][3:0] [[WIRE:.+]] = {{[{}][{}]}}a_0}, {a_1}, {a_2}, {a_3}};
  // CHECK-NEXT: assign b = [[WIRE]][idx];
}

// CHECK-LABEL:   module AddNegLiteral(
// Issue #1324: https://github.com/llvm/circt/issues/1324
hw.module @AddNegLiteral(%a: i8, %x: i8, %y: i8) -> (o1: i8, o2: i8) {

  // CHECK: assign o1 = a - 8'h4;
  %c = hw.constant -4 : i8
  %1 = comb.add %a, %c : i8

  // CHECK: assign o2 = x + y - 8'h4;
  %2 = comb.add %x, %y, %c : i8

  hw.output %1, %2 : i8, i8
}


// CHECK-LABEL:   module ShiftAmountZext(
// Issue #1569: https://github.com/llvm/circt/issues/1569
hw.module @ShiftAmountZext(%a: i8, %b1: i4, %b2: i4, %b3: i4)
 -> (o1: i8, o2: i8, o3: i8) {

  %c = hw.constant 0 : i4
  %B1 = comb.concat %c, %b1 : i4, i4
  %B2 = comb.concat %c, %b2 : i4, i4
  %B3 = comb.concat %c, %b3 : i4, i4

  // CHECK: assign o1 = a << b1;
  %r1 = comb.shl %a, %B1 : i8

  // CHECK: assign o2 = a >> b2;
  %r2 = comb.shru %a, %B2 : i8

  // CHECK: assign o3 = $signed($signed(a) >>> b3);
  %r3 = comb.shrs %a, %B3 : i8
  hw.output %r1, %r2, %r3 : i8, i8, i8
}

// CHECK-LABEL: ModuleWithLocInfo
// CHECK: // Foo.bar:42:13
hw.module @ModuleWithLocInfo()  {
} loc("Foo.bar":42:13)


// CHECK-LABEL: module SignedshiftResultSign
// Issue #1681: https://github.com/llvm/circt/issues/1681
hw.module @SignedshiftResultSign(%a: i18) -> (b: i18) {
  // CHECK: assign b = $signed($signed(a) >>> a[6:0]) ^ 18'hB28;
  %c2856_i18 = hw.constant 2856 : i18
  %c0_i11 = hw.constant 0 : i11
  %0 = comb.extract %a from 0 : (i18) -> i7
  %1 = comb.concat %c0_i11, %0 : i11, i7
  %2 = comb.shrs %a, %1 : i18
  %3 = comb.xor %2, %c2856_i18 : i18
  hw.output %3 : i18
}
// CHECK-LABEL: module SignedShiftRightPrecendence
hw.module @SignedShiftRightPrecendence(%p: i1, %x: i45) -> (o: i45) {
  // CHECK: assign o = $signed($signed(x) >>> (p ? 45'h5 : 45'h8))
  %c5_i45 = hw.constant 5 : i45
  %c8_i45 = hw.constant 8 : i45
  %0 = comb.mux %p, %c5_i45, %c8_i45 : i45
  %1 = comb.shrs %x, %0 : i45
  hw.output %1 : i45
}

// CHECK-LABEL: structExtractChain
hw.module @structExtractChain(%cond: i1, %a: !hw.struct<c: !hw.struct<d:i1>>) -> (out: i1) {
    %1 = hw.struct_extract %a["c"] : !hw.struct<c: !hw.struct<d:i1>>
    %2 = hw.struct_extract %1["d"] : !hw.struct<d:i1>
    // CHECK: assign out = a.c.d;
    hw.output %2 : i1
}

// CHECK-LABEL: structExtractFromTemporary
hw.module @structExtractFromTemporary(%cond: i1, %a: !hw.struct<c: i1>, %b: !hw.struct<c: i1>) -> (out: i1) {
    %0 = comb.mux %cond, %a, %b : !hw.struct<c: i1>
    %1 = hw.struct_extract %0["c"] : !hw.struct<c: i1>
    // CHECK: wire struct packed {logic c; } _GEN = cond ? a : b;
    // CHECK-NEXT: assign out = _GEN.c;
    hw.output %1 : i1
}

// CHECK-LABEL: module unionCreateNoPadding(
// CHECK-NEXT:    input [1:0] in,
// CHECK-NEXT:    output union packed { struct packed {logic a; logic [0:0] __post_padding_a;} a;logic [1:0] b;} out
hw.module @unionCreateNoPadding(%in: i2) -> (out: !hw.union<a: i1, b: i2>) {
  // CHECK: assign out = in + in;
  %add = comb.add %in, %in : i2
  %0 = hw.union_create "b", %add : !hw.union<a: i1, b: i2>
  hw.output %0 : !hw.union<a: i1, b: i2>
}

// CHECK-LABEL: module unionCreatePadding(
// CHECK-NEXT:    input in,
// CHECK-NEXT:    output union packed { struct packed {logic a; logic [0:0] __post_padding_a;} a;logic [1:0] b;} out
hw.module @unionCreatePadding(%in: i1) -> (out: !hw.union<a: i1, b: i2>) {
  // CHECK: assign out = {in, 1'h0};
  %0 = hw.union_create "a", %in : !hw.union<a: i1, b: i2>
  hw.output %0 : !hw.union<a: i1, b: i2>
}

// CHECK-LABEL: module unionCreateZeroWidthElement(
// CHECK-NEXT:    // input /*Zero Width*/ in,
// CHECK-NEXT:    output union packed {/*a: Zero Width;*/ logic [1:0] b;} out
hw.module @unionCreateZeroWidthElement(%in: i0) -> (out: !hw.union<a: i0, b: i2>) {
  // CHECK: assign out = 2'h0;
  %0 = hw.union_create "a", %in : !hw.union<a: i0, b: i2>
  hw.output %0 : !hw.union<a: i0, b: i2>
}

// CHECK-LABEL: unionExtractFromTemporary
hw.module @unionExtractFromTemporary(%cond: i1, %a: !hw.union<c: i1>, %b: !hw.union<c: i1>) -> (out: i1) {
    %0 = comb.mux %cond, %a, %b : !hw.union<c: i1>
    %1 = hw.union_extract %0["c"] : !hw.union<c: i1>
    // CHECK: wire union packed {logic c;} _GEN = cond ? a : b;
    // CHECK-NEXT: assign out = _GEN.c;
    hw.output %1 : i1
}

// CHECK-LABEL: structExplodeLowering
hw.module @structExplodeLowering(%a: !hw.struct<a: i1, b: i1>) -> (outA: i1, outB: i1) {
  // CHECK: assign outA = a.a;
  // CHECK: assign outB = a.b;
  %0:2 = hw.struct_explode %a : !hw.struct<a: i1, b: i1>
  hw.output %0#0, %0#1 : i1, i1
}


// Rename field names
// CHECK-LABEL: renameKeyword(
// CHECK-NEXT:  input  struct packed {logic repeat_0; logic repeat_0_0; } a,
// CHECK-NEXT:  output struct packed {logic repeat_0; logic repeat_0_0; } r1
// CHECK-NEXT:  );
hw.module @renameKeyword(%a: !hw.struct<repeat: i1, repeat_0: i1>) -> (r1: !hw.struct<repeat: i1, repeat_0: i1>){
  hw.output %a : !hw.struct<repeat: i1, repeat_0: i1>
}

// CHECK-LABEL: useRenamedStruct(
// CHECK-NEXT:  inout  struct packed {logic repeat_0; logic repeat_0_0; } a,
// CHECK-NEXT:  output                                                    r1,
// CHECK-NEXT:                                                            r2,
// CHECK-NEXT:  output struct packed {logic repeat_0; logic repeat_0_0; } r3, 
// CHECK-NEXT:                                                            r4
// CHECK-NEXT:  );
hw.module @useRenamedStruct(%a: !hw.inout<struct<repeat: i1, repeat_0: i1>>) -> (r1: i1, r2: i1, r3: !hw.struct<repeat: i1, repeat_0: i1>, r4: !hw.struct<repeat: i1, repeat_0: i1>) {
  %read = sv.read_inout %a : !hw.inout<struct<repeat: i1, repeat_0: i1>>

  %i0 = hw.instance "inst1" @renameKeyword(a: %read: !hw.struct<repeat: i1, repeat_0: i1>) -> (r1: !hw.struct<repeat: i1, repeat_0: i1>)
  // CHECK:      renameKeyword inst1 (
  // CHECK-NEXT:   .a  (a),
  // CHECK-NEXT:   .r1 (r4)
  // CHECK-NEXT: )

  %0 = sv.struct_field_inout %a["repeat"] : !hw.inout<struct<repeat: i1, repeat_0: i1>>
  %1 = sv.read_inout %0 : !hw.inout<i1>
  // assign r1 = a.repeat_0;
  %2 = hw.struct_extract %read["repeat_0"] : !hw.struct<repeat: i1, repeat_0: i1>
  // assign r2 = a.repeat_0_1;
  %true = hw.constant true
  %3 = hw.struct_inject %read["repeat_0"], %true : !hw.struct<repeat: i1, repeat_0: i1>
  // assign r3 = '{repeat_0: a.repeat_0, repeat_0_0: (1'h1)};
  hw.output %1, %2, %3, %i0 : i1, i1, !hw.struct<repeat: i1, repeat_0: i1>, !hw.struct<repeat: i1, repeat_0: i1>
}


// CHECK-LABEL: module replicate
hw.module @replicate(%arg0: i7, %arg1: i1) -> (r1: i21, r2: i9, r3: i16, r4: i16) {
  // CHECK: assign r1 = {3{arg0}};
  %r1 = comb.replicate %arg0 : (i7) -> i21

  // CHECK: assign r2 = {9{arg1}};
  %r2 = comb.replicate %arg1 : (i1) -> i9

  // CHECK: assign r3 = {{[{]}}{9{arg0[6]}}, arg0};
  %0 = comb.extract %arg0 from 6 : (i7) -> i1
  %1 = comb.replicate %0 : (i1) -> i9
  %r3 = comb.concat %1, %arg0 : i9, i7

  // CHECK: assign r4 = {2{arg0, arg1}};
  %2 = comb.concat %arg0, %arg1 : i7, i1
  %r4 = comb.replicate %2 : (i8) -> i16

  hw.output %r1, %r2, %r3, %r4 : i21, i9, i16, i16
}

// CHECK-LABEL: module addParenthesesToSuccessiveOperators
hw.module @addParenthesesToSuccessiveOperators(%a: i4, %b: i1, %c: i4) -> (o1:i1, o2:i1, o3:i1) {
  %one4 = hw.constant -1 : i4
  %zero4 = hw.constant 0 : i4
  // CHECK: wire [[GEN:.+]] = &c;

  %and1 = comb.icmp eq %a, %one4 : i4
  %and2 = comb.icmp eq %a, %one4 : i4
  %and3 = comb.icmp eq %a, %one4 : i4
  %and = comb.and %and1, %b, %and2, %and3  : i1
  // CHECK-NEXT: assign o1 = (&a) & b & (&a) & (&a);

  %or1 = comb.icmp ne %a, %zero4 : i4
  %or2 = comb.icmp ne %a, %zero4 : i4
  %or3 = comb.icmp ne %a, %zero4 : i4
  %or = comb.or %or1, %b, %or2, %or3  : i1
  // CHECK-NEXT: assign o2 = (|a) | b | (|a) | (|a);

  %3 = comb.icmp eq %c, %one4 : i4
  %multiuse = comb.and %3, %3 : i1
  // CHECK-NEXT: assign o3 = [[GEN]] & [[GEN]];

  hw.output %and, %or, %multiuse : i1, i1, i1
}

// CHECK-LABEL: module parameters
// CHECK-NEXT: #(parameter [41:0] p1 = 42'd17
// CHECK-NEXT:   parameter [0:0]  p2) (
// CHECK-NEXT: input  [7:0] arg0,
hw.module @parameters<p1: i42 = 17, p2: i1>(%arg0: i8) -> (out: i8) {
  // Local values should not conflict with output or parameter names.
  // CHECK: wire [3:0] p1_0;
  %p1 = sv.wire : !hw.inout<i4>

  %out = sv.wire : !hw.inout<i4>
  // CHECK: wire [3:0] out_0;
  hw.output %arg0 : i8
}

hw.module.extern @parameters2<p1: i42 = 17, p2: i1 = 0>(%arg0: i8) -> (out: i8)

// CHECK-LABEL: module parameters3
// CHECK-NEXT:   #(parameter [41:0] p1 = 42'd17) (
// CHECK-NEXT:   input  [p1 - 1:0] arg0,
// CHECK-NEXT:   output [p1 - 1:0] out
// CHECK-NEXT:  );
// CHECK:   assign out = arg0;
hw.module @parameters3<p1: i42 = 17>(%arg0: !hw.int<#hw.param.decl.ref<"p1">>) -> (out: !hw.int<#hw.param.decl.ref<"p1">>) {
  hw.output %arg0 : !hw.int<#hw.param.decl.ref<"p1">>
}

// CHECK-LABEL: module UseParameterized(
hw.module @UseParameterized(%a: i8) -> (ww: i8, xx: i8, yy: i8, zz: i8, qq: i8) {
  // Two parameters.
  // CHECK:      parameters #(
  // CHECK-NEXT:   .p1(42'd4),
  // CHECK-NEXT:   .p2(0)
  // CHECK-NEXT: ) inst1 (
  // CHECK-NEXT:   .arg0 (a),
  // CHECK-NEXT:   .out  (ww)
  // CHECK-NEXT: );
  %r0 = hw.instance "inst1" @parameters<p1: i42 = 4, p2: i1 = 0>(arg0: %a: i8) -> (out: i8)

  // Two parameters.
  // CHECK:      parameters #(
  // CHECK-NEXT:   .p1(42'd11),
  // CHECK-NEXT:   .p2(1)
  // CHECK-NEXT: ) inst2 (
  // CHECK-NEXT:   .arg0 (a),
  // CHECK-NEXT:   .out  (xx)
  // CHECK-NEXT: );
  %r1 = hw.instance "inst2" @parameters<p1: i42 = 11, p2: i1 = 1>(arg0: %a: i8) -> (out: i8)

  // One default, don't print it
  // CHECK:      parameters #(
  // CHECK-NEXT:   .p2(0)
  // CHECK-NEXT: ) inst3 (
  // CHECK-NEXT:   .arg0 (a),
  // CHECK-NEXT:   .out  (yy)
  // CHECK-NEXT: );
  %r2 = hw.instance "inst3" @parameters<p1: i42 = 17, p2: i1 = 0>(arg0: %a: i8) -> (out: i8)

  // All defaults, don't print a parameter list at all.
  // CHECK:      parameters2 inst4 (
  // CHECK-NEXT:   .arg0 (a),
  // CHECK-NEXT:   .out  (zz)
  // CHECK-NEXT: );
  %r3 = hw.instance "inst4" @parameters2<p1: i42 = 17, p2: i1 = 0>(arg0: %a: i8) -> (out: i8)

  // Parameterized I/O ports.
  // CHECK: parameters3 #(
  // CHECK-NEXT:   .p1(42'd8)
  // CHECK-NEXT: ) inst5 (
  // CHECK-NEXT:   .arg0 (a),
  // CHECK-NEXT:   .out  (qq)
  // CHECK-NEXT: );
  %r4 = hw.instance "inst5" @parameters3<p1: i42 = 8>(arg0: %a: i8) -> (out: i8)

  hw.output %r0, %r1, %r2, %r3, %r4: i8, i8, i8, i8, i8
}

// CHECK-LABEL: module UseParameterValue
hw.module @UseParameterValue<xx: i42>(%arg0: i8)
  -> (out1: i8, out2: i8, out3: i8, out4: i42) {
  // CHECK-NEXT: #(parameter [41:0] xx) (

  // CHECK:      parameters2 #(
  // CHECK-NEXT:  .p1(xx)
  // CHECK-NEXT: ) inst1 (
  %a = hw.instance "inst1" @parameters2<p1: i42 = #hw.param.decl.ref<"xx">, p2: i1 = 0>(arg0: %arg0: i8) -> (out: i8)

  // CHECK:      parameters2 #(
  // CHECK-NEXT:  .p1(xx + 42'd17)
  // CHECK-NEXT: ) inst2 (
  %b = hw.instance "inst2" @parameters2<p1: i42 = #hw.param.expr.add<#hw.param.verbatim<"xx">, 17>, p2: i1 = 0>(arg0: %arg0: i8) -> (out: i8)

  // CHECK:      parameters2 #(
  // CHECK-NEXT:  .p1(xx * yy + yy * 42'd17)
  // CHECK-NEXT: ) inst3 (
  %c = hw.instance "inst3" @parameters2<p1: i42 = #hw.param.expr.mul<#hw.param.expr.add<#hw.param.verbatim<"xx">, 17>, #hw.param.verbatim<"yy">>, p2: i1 = 0>(arg0: %arg0: i8) -> (out: i8)

  // FIXME: Decl word should be localparam.
  // CHECK: wire [41:0] _GEN = xx + 42'd17;
  // CHECK-NEXT: assign out3 = _GEN[7:0] + _GEN[7:0];
  %d = hw.param.value i42 = #hw.param.expr.add<#hw.param.decl.ref<"xx">, 17>
  %e = comb.extract %d from 0 : (i42) -> i8
  %f = comb.add %e, %e : i8

  // CHECK-NEXT: assign out4 = $signed(42'd4) >>> $signed(xx);
  %g = hw.param.value i42 = #hw.param.expr.shrs<4, #hw.param.decl.ref<"xx">>

  hw.output %a, %b, %f, %g : i8, i8, i8, i42
}

// CHECK-LABEL: module VerilogCompatParameters
hw.module @VerilogCompatParameters<p1: i42, p2: i32, p3: f64 = 1.5,
                                   p4: i32 = 4, p5: none = "foo",
                                   p6: none>()
  -> () {
  // CHECK-NEXT: #(parameter [41:0]      p1,
  // CHECK-NEXT:   parameter /*integer*/ p2,
  // CHECK-NEXT:   parameter             p3 = 1.500000e+00,
  // CHECK-NEXT:   parameter             p4 = 4,
  // CHECK-NEXT:   parameter             p5 = "foo",
  // CHECK-NEXT:   parameter             p6)

}


// CHECK-LABEL: module parameterizedTypes
// CHECK: #(parameter param = 1,
// CHECK:   parameter wire_0 = 2) (
hw.module @parameterizedTypes<param: i32 = 1, wire: i32 = 2>
  // CHECK: input [16:0]{{ *}}a,
  (%a: !hw.int<17>,
  // CHECK: input [param - 1:0]{{ *}}b
   %b: !hw.int<#hw.param.decl.ref<"param">>,
  // CHECK: input [$clog2($unsigned(param)) - 1:0]{{ *}}c
   %c: !hw.int<#hw.param.expr.clog2<#hw.param.decl.ref<"param">>>) {

  // Check that the parameter name renamification propagates.
  // CHECK: wire [wire_0 - 1:0] paramWire;
  %paramWire = sv.wire : !hw.inout<!hw.int<#hw.param.decl.ref<"wire">>>

}

// CHECK-LABEL: // moduleWithComment has a comment
// CHECK-NEXT:  // hello
// CHECK-NEXT:  module moduleWithComment
hw.module @moduleWithComment()
  attributes {comment = "moduleWithComment has a comment\nhello"} {}


// CHECK-LABEL: module Foo(
// https://github.com/llvm/circt/issues/2363
hw.module @Foo(%a: i1, %b: i1) -> (r1: i1, r2: i1) {
  // Make sure the temporary wire is indented correctly.
  // CHECK: {{^  wire _GEN = a == b;}}
  %0 = comb.icmp eq %a, %b : i1
  hw.output %0, %0 : i1, i1
}

// CHECK-LABEL: module parameterizedArrays
// CHECK-NEXT:   #(parameter /*integer*/ param,
// CHECK-NEXT:     parameter /*integer*/ N) (
// CHECK-NEXT:   input  [41:0][param - 1:0]        a,
// CHECK-NEXT:   input  [N - 64'd1:0][param - 1:0] b,
// CHECK-NEXT:   output [N - 64'd1:0][param - 1:0] c
// CHECK-NEXT:  );
hw.module @parameterizedArrays<param: i32, N: i32>
  (%a: !hw.array<42x!hw.int<#hw.param.decl.ref<"param">>>,
   %b: !hw.array<#hw.param.decl.ref<"N"> x !hw.int<#hw.param.decl.ref<"param">>>) ->
   (c: !hw.array<#hw.param.decl.ref<"N"> x !hw.int<#hw.param.decl.ref<"param">>>) {
  hw.output %b : !hw.array<#hw.param.decl.ref<"N"> x !hw.int<#hw.param.decl.ref<"param">>>
}

// CHECK-LABEL: module UseParameterizedArrays(
// CHECK-NEXT: input [41:0][11:0] a,
// CHECK-NEXT: input [23:0][11:0] b
// CHECK-NEXT: output [23:0][11:0] c
// CHECK-NEXT: );
hw.module @UseParameterizedArrays(%a: !hw.array<42xint<12>>, %b: !hw.array<24xint<12>>) -> (c: !hw.array<24xint<12>>) {
// CHECK:  parameterizedArrays #(
// CHECK-NEXT:    .param(12),
// CHECK-NEXT:    .N(24)
// CHECK-NEXT:  ) inst (
// CHECK-NEXT:    .a (a),
// CHECK-NEXT:    .b (b),
// CHECK-NEXT:    .c (c)
// CHECK-NEXT:  );
// CHECK-NEXT: endmodule
  %c = hw.instance "inst" @parameterizedArrays<param: i32 = 12, N: i32 = 24>
    (a: %a : !hw.array<42xint<12>>, b: %b : !hw.array<24xint<12>>) -> (c: !hw.array<24xint<12>>) {}
  hw.output %c: !hw.array<24xint<12>>
}

// CHECK-LABEL: module NoneTypeParam
// CHECK:         #(parameter p1) ();
// CHECK:       endmodule
hw.module @NoneTypeParam<p1: none>() -> () {}

// CHECK-LABEL: module ParamConcatInst
// CHECK:         #(parameter name = "top") ();
// CHECK:         NoneTypeParam #(
// CHECK:           .p1({".", name, ".child"})
// CHECK:         ) inst ();
// CHECK:       endmodule
hw.module @ParamConcatInst<name: none = "top">() -> () {
  hw.instance "inst" @NoneTypeParam<p1: none = #hw.param.expr.str.concat<".", #hw.param.decl.ref<"name">, ".", "child">>() -> ()
}

// CHECK-LABEL: module ParamsParensPrecedence
hw.module @ParamsParensPrecedence<param: i32>() -> (a:i32, b:i32, c:i32) {
  // CHECK: = $clog2($unsigned(param));
  %1 = hw.param.value i32 = #hw.param.expr.clog2<#hw.param.decl.ref<"param">>

  // CHECK: = $clog2($unsigned($clog2($unsigned(param + 8))));
  %3 = hw.param.value i32 = #hw.param.expr.clog2<#hw.param.expr.clog2<#hw.param.expr.add<#hw.param.decl.ref<"param">,8>>>

  // CHECK: = $signed(param) >>> $signed(param & 8);
  %4 = hw.param.value i32 = #hw.param.expr.shrs<#hw.param.decl.ref<"param">,#hw.param.expr.and<8,#hw.param.decl.ref<"param">>>
  hw.output %1, %3, %4: i32, i32, i32
}

// CHECK-LABEL: module ArrayGetInline
hw.module @ArrayGetInline(%a: !hw.array<4xstruct<a: i32>>, %b: !hw.array<4xi1>, %idx: i2, %idx_port: !hw.inout<i2>)
                          -> (out: i32, out2: i1, out3: i1, out4: i1, out5: i1) {
  %c0_i2 = hw.constant 0 : i2
  %x = hw.array_get %a[%c0_i2] : !hw.array<4xstruct<a: i32>>, i2
  %y = hw.struct_extract %x["a"] : !hw.struct<a: i32>
  // CHECK:      assign out = a[2'h0].a;
  // CHECK-NEXT: assign out2 = b[idx];
  // CHECK-NEXT: assign out3 = b[idx];
  // CHECK-NEXT: assign out4 = b[idx_port];
  // CHECK-NEXT: assign out5 = b[idx_port];
  %array_get_idx = hw.array_get %b[%idx] : !hw.array<4xi1>, i2
  %read = sv.read_inout %idx_port : !hw.inout<i2>
  %array_get_idx_port = hw.array_get %b[%read] : !hw.array<4xi1>, i2
  hw.output %y, %array_get_idx, %array_get_idx, %array_get_idx_port, %array_get_idx_port : i32, i1, i1, i1, i1
}

// CHECK-LABEL: module UniformArrayCreate
hw.module @UniformArrayCreate() -> (arr: !hw.array<5xi8>) {
  %c0_i8 = hw.constant 0 : i8
  %arr = hw.array_create %c0_i8, %c0_i8, %c0_i8, %c0_i8, %c0_i8 : i8
  // CHECK: assign arr = {5{8'h0}};
  hw.output %arr : !hw.array<5xi8>
}

// CHECK-LABEL: module Issue4485(
// CHECK-NEXT:    input [3:0] in
// CHECK-NEXT:  );
// CHECK-EMPTY:
// CHECK-NEXT:  endmodule
hw.module @Issue4485(%in: i4) {
  %c0_i4 = hw.constant 0 : i4
  %1 = comb.icmp eq %in, %c0_i4 : i4
  %2 = sv.system.sampled %1 : i1
  hw.output
}

// CHECK-LABEL: module inline_bitcast_in_concat(
// CHECK-NEXT:    input  [6:0]      in1,
// CHECK-NEXT:    input  [7:0][3:0] in2,
// CHECK-NEXT:    output [38:0]     out
// CHECK-NEXT:  );
// CHECK-EMPTY:
// CHECK-NEXT:    assign out = {in1, /*cast(bit[31:0])*/in2};
// CHECK-NEXT:  endmodule
hw.module @inline_bitcast_in_concat(%in1: i7, %in2: !hw.array<8xi4>) -> (out: i39) {
  %r2 = hw.bitcast %in2 : (!hw.array<8xi4>) -> i32
  %0 = comb.concat %in1, %r2: i7, i32
  hw.output %0 : i39
}

// CHECK-LABEL: module DontInlineAggregateConstantIntoPorts(
// CHECK:         wire [1:0][3:0] _GEN = {4'h0, 4'h1};
// CHECK-NEXT:    Array i0 (
// CHECK-NEXT:     .a (_GEN)
// CHECK-NEXT:    );
// CHECK-NEXT:  endmodule
hw.module @DontInlineAggregateConstantIntoPorts() -> () {
  %0 = hw.aggregate_constant [0 : i4, 1 : i4] : !hw.array<2xi4>
  hw.instance "i0" @Array(a: %0: !hw.array<2xi4>) -> ()
}

// CHECK-LABEL: module FooA(
// CHECK-NEXT:    input union packed {logic [15:0] a; struct packed {logic [9:0] b; logic [5:0] __post_padding_b;} b;} test
// CHECK-NEXT:    output [15:0] a,
// CHECK-NEXT:    output [9:0] b
// CHECK-NEXT:  );
// CHECK-EMPTY:
// CHECK-NEXT:    assign a = test.a;
// CHECK-NEXT:    assign b = test.b.b;
// CHECK-NEXT:  endmodule
!unionA = !hw.union<a: i16, b: i10>
hw.module @FooA(%test: !unionA) -> (a: i16, b: i10) {
  %0 = hw.union_extract %test["a"] : !unionA
  %1 = hw.union_extract %test["b"] : !unionA
  hw.output %0, %1 : i16, i10
}

// CHECK-LABEL: module FooB(
// CHECK-NEXT:    input union packed {logic [15:0] a; struct packed {logic [1:0] __pre_padding_b; logic [13:0] b;} b;} test,
// CHECK-NEXT:    output [15:0] a,
// CHECK-NEXT:    output [13:0] b
// CHECK-NEXT:  );
// CHECK-EMPTY:
// CHECK-NEXT:    assign a = test.a;
// CHECK-NEXT:    assign b = test.b.b;
// CHECK-NEXT:  endmodule
!unionB = !hw.union<a: i16, b: i14 offset 2>
hw.module @FooB(%test: !unionB) -> (a: i16, b: i14) {
  %0 = hw.union_extract %test["a"] : !unionB
  %1 = hw.union_extract %test["b"] : !unionB
  hw.output %0, %1 : i16, i14
}
