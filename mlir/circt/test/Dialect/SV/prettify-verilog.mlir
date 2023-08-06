// RUN: circt-opt -prettify-verilog %s | FileCheck %s
// RUN: circt-opt -prettify-verilog %s | circt-opt --export-verilog | FileCheck %s --check-prefix=VERILOG

// CHECK-LABEL: hw.module @unary_ops
hw.module @unary_ops(%arg0: i8, %arg1: i8, %arg2: i8, %arg3: i1)
   -> (a: i8, b: i8, c: i1) {
  %c-1_i8 = hw.constant -1 : i8

  // CHECK: [[XOR1:%.+]] = comb.xor %arg0
  %unary = comb.xor %arg0, %c-1_i8 : i8
  // CHECK: %1 = comb.add [[XOR1]], %arg1
  %a = comb.add %unary, %arg1 : i8

  // CHECK: [[XOR2:%.+]] = comb.xor %arg0
  // CHECK: %3 = comb.add [[XOR2]], %arg2
  %b = comb.add %unary, %arg2 : i8


  // Multi-use arith.xori gets duplicated, and we need to make sure there is a local
  // constant as well.
  %true = hw.constant true
  %c = comb.xor %arg3, %true : i1

  // CHECK: [[TRUE1:%.+]] = hw.constant true
  sv.initial {
    // CHECK: [[TRUE2:%.+]] = hw.constant true
    // CHECK: [[XOR3:%.+]] = comb.xor %arg3, [[TRUE2]]
    // CHECK: sv.if [[XOR3]]
    sv.if %c {
      sv.fatal 1
    }
  }

  // CHECK: [[XOR4:%.+]] = comb.xor %arg3, [[TRUE1]]
  // CHECK: hw.output %1, %3, [[XOR4]]
  hw.output %a, %b, %c : i8, i8, i1
}

// VERILOG: assign a = ~arg0 + arg1;
// VERILOG: assign b = ~arg0 + arg2;


/// The pass should sink constants in to the block where they are used.
// CHECK-LABEL: @sink_constants
// VERILOG-LABEL: sink_constants
hw.module @sink_constants(%clock :i1) -> (out : i1){
  // CHECK: %false = hw.constant false
  %false = hw.constant false

  // CHECK-NOT: %fd = hw.constant -2147483646 : i32
  %fd = hw.constant 0x80000002 : i32

  /// Constants not used should be removed.
  // CHECK-NOT: %true = hw.constant true
  %true = hw.constant true

  /// Simple constant sinking.
  sv.ifdef "FOO" {
    sv.initial {
      // CHECK: [[FALSE:%.*]] = hw.constant false
      // CHECK: [[FD:%.*]] = hw.constant -2147483646 : i32
      // CHECK: [[TRUE:%.*]] = hw.constant true
      // CHECK: sv.fwrite [[FD]], "%x"([[TRUE]]) : i1
      sv.fwrite %fd, "%x"(%true) : i1
      // CHECK: sv.fwrite [[FD]], "%x"([[FALSE]]) : i1
      sv.fwrite %fd, "%x"(%false) : i1
    }
  }

  /// Multiple uses in the same block should use the same constant.
  sv.ifdef "FOO" {
    sv.initial {
      // CHECK: [[FD:%.*]] = hw.constant -2147483646 : i32
      // CHECK: [[TRUE:%.*]] = hw.constant true
      // CHECK: sv.fwrite [[FD]], "%x"([[TRUE]]) : i1
      // CHECK: sv.fwrite [[FD]], "%x"([[TRUE]]) : i1
      sv.fwrite %fd, "%x"(%true) : i1
      sv.fwrite %fd, "%x"(%true) : i1
    }
  }

  // CHECK: hw.output %false : i1
  hw.output %false : i1
}

// VERILOG: `ifdef FOO
// VERILOG:   $fwrite(32'h80000002, "%x", 1'h1);
// VERILOG:   $fwrite(32'h80000002, "%x", 1'h0);
// VERILOG: `endif
// VERILOG: `ifdef FOO
// VERILOG:   $fwrite(32'h80000002, "%x", 1'h1);
// VERILOG:   $fwrite(32'h80000002, "%x", 1'h1);
// VERILOG: `endif

// Prettify should always sink ReadInOut to its usage.
// CHECK-LABEL: @sinkReadInOut
// VERILOG-LABEL: sinkReadInOut
hw.module @sinkReadInOut(%clk: i1) {
  %myreg = sv.reg  : !hw.inout<array<1xstruct<a: i48, b: i48>>>
  %false = hw.constant false
  %0 = sv.array_index_inout %myreg[%false]: !hw.inout<array<1xstruct<a: i48, b: i48>>>, i1
  %a = sv.struct_field_inout %0["a"]: !hw.inout<struct<a: i48, b: i48>>
  %b = sv.struct_field_inout %0["b"]: !hw.inout<struct<a: i48, b: i48>>
  %2 = sv.read_inout %b : !hw.inout<i48>
  sv.alwaysff(posedge %clk)  {
    sv.passign %a, %2 : i48
  }
}
// CHECK:  %myreg = sv.reg
// CHECK:  sv.alwaysff(posedge %clk)
// CHECK:    sv.array_index_inout
// CHECK:    sv.struct_field_inout
// CHECK:    sv.read_inout

// VERILOG:  struct packed {logic [47:0] a; logic [47:0] b; }[0:0] myreg;
// VERILOG:  always_ff @(posedge clk)
// VERILOG:    myreg[1'h0].a <= myreg[1'h0].b;


// CHECK-LABEL: @sink_expression
// VERILOG-LABEL: sink_expression
hw.module @sink_expression(%clock: i1, %a: i1, %a2: i1, %a3: i1, %a4: i1) {
  // This or is used in one place.
  %0 = comb.or %a2, %a3 : i1
  // This and/xor chain is used in two.  Both should be sunk.
  %1 = comb.and %a2, %a3 : i1
  %2 = comb.xor %1, %a4 : i1
  // CHECK: sv.always
  sv.always posedge %clock  {
    // CHECK: [[AND:%.*]] = comb.and %a2, %a3 : i1
    // CHECK: [[XOR:%.*]] = comb.xor [[AND]], %a4 : i1

    // CHECK: sv.ifdef.procedural
    sv.ifdef.procedural "SOMETHING"  {
      // CHECK: [[OR:%.*]] = comb.or %a2, %a3 : i1
      // CHECK: sv.if [[OR]]
      sv.if %0  {
        sv.fatal 1
      }
      // CHECK: sv.if [[XOR]]
      sv.if %2  {
        sv.fatal 1
      }
    }

    // CHECK: sv.if [[XOR]]
    sv.if %2  {
      sv.fatal 1
    }
  }
  hw.output
}

// CHECK-LABEL: @dont_sink_se_expression
hw.module @dont_sink_se_expression(%clock: i1, %a: i1, %a2: i1, %a3: i1, %a4: i1) {

  // CHECK: [[DONT_TOUCH:%.*]] = sv.verbatim.expr.se "DONT_TOUCH"
  %0 = sv.verbatim.expr "SINK_ME" : () -> i1
  %1 = sv.verbatim.expr.se "DONT_TOUCH" : () -> i1

  // CHECK: sv.always
  sv.always posedge %clock  {
    // CHECK: [[SINK:%.*]] = sv.verbatim.expr "SINK_ME"
    // CHECK: sv.if [[SINK]]
    sv.if %0  {
      sv.fatal 1
    }

    // CHECK: sv.if [[DONT_TOUCH]]
    sv.if %1  {
      sv.fatal 1
    }
  }
  hw.output
}

hw.module.extern @MyExtModule(%in: i8)

// CHECK-LABEL: hw.module @MoveInstances
// VERILOG-LABEL: module MoveInstances
hw.module @MoveInstances(%a_in: i8) {
  // CHECK: %0 = comb.add %a_in, %a_in : i8
  // CHECK: hw.instance "xyz3" @MyExtModule(in: %0: i8)
  // VERILOG: MyExtModule xyz3 (
  // VERILOG:   .in (a_in + a_in)
  // VERILOG: );
  hw.instance "xyz3" @MyExtModule(in: %b: i8) -> ()

  %b = comb.add %a_in, %a_in : i8
}


// CHECK-LABEL: hw.module @unary_sink_crash
hw.module @unary_sink_crash(%arg0: i1) {
  %true = hw.constant true
  %c = comb.xor %arg0, %true : i1
  // CHECK-NOT: hw.constant
  // CHECK-NOT: comb.xor
  // CHECK: sv.initial
  sv.initial {
    // CHECK: [[TRUE1:%.+]] = hw.constant true
    // CHECK: [[XOR1:%.+]] = comb.xor %arg0, [[TRUE1]]
    // CHECK: sv.if [[XOR1]]
    sv.if %c {
      sv.fatal 1
    }

    // CHECK: [[TRUE2:%.+]] = hw.constant true
    // CHECK: [[XOR2:%.+]] = comb.xor %arg0, [[TRUE2]]
    // CHECK: sv.if [[XOR2]]
    sv.if %c {
      sv.fatal 1
    }
  }
}


// CHECK-LABEL: hw.module @unary_sink_no_duplicate
// https://github.com/llvm/circt/issues/2097
hw.module @unary_sink_no_duplicate(%arg0: i4) -> (result: i4) {
  %ones = hw.constant 15: i4

  // CHECK-NOT: comb.xor

  // We normally duplicate unary operations like this one so they can be inlined
  // into the using expressions.  However, not all users can be inlined *into*.
  // Things like extract/sext do not support this, so do not duplicate if used
  // by one of them.

  // CHECK: comb.xor %arg0,
  %0 = comb.xor %arg0, %ones : i4

 // CHECK-NOT: comb.xor

  %a = comb.extract %0 from 0 : (i4) -> i1
  %b = comb.extract %0 from 1 : (i4) -> i1
  %c = comb.extract %0 from 2 : (i4) -> i2


  // CHECK: hw.output
  %r = comb.concat %a, %b, %c : i1, i1, i2
  hw.output %r : i4
}

// CHECK-LABEL: hw.module private @ConnectToAllFields
hw.module private @ConnectToAllFields(%clock: i1, %reset: i1, %value: i2, %base: !hw.inout<!hw.struct<a: i2>>) -> () {
  %r = sv.reg : !hw.inout<!hw.struct<a: i2>>
  %val = sv.read_inout %r : !hw.inout<!hw.struct<a: i2>>
  sv.always posedge %clock {
    sv.passign %r, %1 : !hw.struct<a: i2>
  }

  %0 = sv.read_inout %base : !hw.inout<!hw.struct<a: i2>>
  %1 = hw.struct_inject %0["a"], %value : !hw.struct<a: i2>
  // CHECK: sv.always posedge %clock {
  // CHECK:   [[FIELD_A:%.+]] = sv.struct_field_inout %r["a"] : !hw.inout<struct<a: i2>>
  // CHECK:   sv.passign [[FIELD_A]], %value : i2
  // CHECK: }

  // VERILOG:       always @(posedge clock)
  // VERILOG-NEXT:    r.a <= value;
}

// CHECK-LABEL: hw.module private @ConnectSubfield
hw.module private @ConnectSubfield(%clock: i1, %reset: i1, %value: i2) -> () {
  %r = sv.reg : !hw.inout<!hw.struct<a: i2>>
  %val = sv.read_inout %r : !hw.inout<!hw.struct<a: i2>>
  sv.always posedge %clock {
    sv.passign %r, %0 : !hw.struct<a: i2>
  }

  %0 = hw.struct_inject %val["a"], %value : !hw.struct<a: i2>
  // CHECK: sv.always posedge %clock {
  // CHECK:   [[FIELD_A:%.+]] = sv.struct_field_inout %r["a"] : !hw.inout<struct<a: i2>>
  // CHECK:   sv.passign [[FIELD_A]], %value : i2
  // CHECK: }

  //VERILOG:      always @(posedge clock)
  //VERILOG-NEXT:   r.a <= value;
}

// CHECK-LABEL: hw.module private @ConnectSubfields
hw.module private @ConnectSubfields(%clock: i1, %reset: i1, %value2: i2, %value3: i3) -> () {
  %r = sv.reg : !hw.inout<!hw.struct<a: i2, b: i3>>
  %val = sv.read_inout %r : !hw.inout<!hw.struct<a: i2, b: i3>>
  sv.always posedge %clock {
    sv.passign %r, %1 : !hw.struct<a: i2, b: i3>
  }

  %0 = hw.struct_inject %val["a"], %value2 : !hw.struct<a: i2, b: i3>
  %1 = hw.struct_inject %0["b"], %value3 : !hw.struct<a: i2, b: i3>
  // CHECK: sv.always posedge %clock {
  // CHECK:   [[FIELD_A:%.+]] = sv.struct_field_inout %r["a"] : !hw.inout<struct<a: i2, b: i3>>
  // CHECK:   sv.passign [[FIELD_A]], %value2 : i2
  // CHECK:   [[FIELD_B:%.+]] = sv.struct_field_inout %r["b"] : !hw.inout<struct<a: i2, b: i3>>
  // CHECK:   sv.passign [[FIELD_B]], %value3 : i3
  // CHECK: }

  // VERILOG:      always @(posedge clock) begin
  // VERILOG-NEXT:   r.a <= value2;
  // VERILOG-NEXT:   r.b <= value3;
  // VERILOG-NEXT: end
}

// CHECK-LABEL: hw.module private @ConnectSubfieldOverwrite
hw.module private @ConnectSubfieldOverwrite(%clock: i1, %reset: i1, %value2: i2, %value3: i2) -> () {
  %r = sv.reg : !hw.inout<!hw.struct<a: i2, b: i3>>
  %val = sv.read_inout %r : !hw.inout<!hw.struct<a: i2, b: i3>>
  sv.always posedge %clock {
    sv.passign %r, %1 : !hw.struct<a: i2, b: i3>
  }

  %0 = hw.struct_inject %val["a"], %value2 : !hw.struct<a: i2, b: i3>
  %1 = hw.struct_inject %0["a"], %value3 : !hw.struct<a: i2, b: i3>
  // CHECK: sv.always posedge %clock {
  // CHECK:   [[FIELD:%.+]] = sv.struct_field_inout %r["a"] : !hw.inout<struct<a: i2, b: i3>>
  // CHECK:   sv.passign [[FIELD]], %value3 : i2
  // CHECK: }

  // VERILOG:      always @(posedge clock)
  // VERILOG-NEXT:   r.a <= value3;
}

// CHECK-LABEL: hw.module private @ConnectNestedSubfield
hw.module private @ConnectNestedSubfield(%clock: i1, %reset: i1, %value: i2) -> () {
  %r = sv.reg : !hw.inout<!hw.struct<a: !hw.struct<b: i2>>>
  %val = sv.read_inout %r : !hw.inout<!hw.struct<a: !hw.struct<b: i2>>>
  sv.always posedge %clock {
    sv.passign %r, %2 : !hw.struct<a: !hw.struct<b: i2>>
  }
  %0 = hw.struct_extract %val["a"] : !hw.struct<a: !hw.struct<b: i2>>
  %1 = hw.struct_inject %0["b"], %value : !hw.struct<b: i2>
  %2 = hw.struct_inject %val["a"], %1 : !hw.struct<a: !hw.struct<b: i2>>

  // CHECK: sv.always posedge %clock {
  // CHECK:   [[FIELD_A:%.+]] = sv.struct_field_inout %r["a"] : !hw.inout<struct<a: !hw.struct<b: i2>>>
  // CHECK:   [[FIELD_B:%.+]] = sv.struct_field_inout [[FIELD_A]]["b"] : !hw.inout<struct<b: i2>>
  // CHECK:   sv.passign [[FIELD_B]], %value : i2
  // CHECK: }

  // VERILOG:      always @(posedge clock)
  // VERILOG-NEXT:   r.a.b <= value;
}


// CHECK-LABEL: hw.module private @ConnectSubindexMid
hw.module private @ConnectSubindexMid(%clock: i1, %reset: i1, %value: i2) -> () {
  %c0_i2 = hw.constant 0 : i2
  %c-2_i2 = hw.constant -2 : i2
  %r = sv.reg : !hw.inout<!hw.array<3xi2>>
  %val = sv.read_inout %r : !hw.inout<!hw.array<3xi2>>
  sv.always posedge %clock {
    sv.passign %r, %3 : !hw.array<3xi2>
  }

  %0 = hw.array_slice %val[%c0_i2] : (!hw.array<3xi2>) -> !hw.array<1xi2>
  %1 = hw.array_create %value : i2
  %2 = hw.array_slice %val[%c-2_i2] : (!hw.array<3xi2>) -> !hw.array<1xi2>
  %3 = hw.array_concat %2, %1, %0 : !hw.array<1xi2>, !hw.array<1xi2>, !hw.array<1xi2>
  // CHECK: sv.always posedge %clock {
  // CHECK:   [[FIELD:%.+]] = sv.array_index_inout %r[%c1_i2] : !hw.inout<array<3xi2>>, i2
  // CHECK:   sv.passign [[FIELD]], %value : i2
  // CHECK: }

  // VERILOG:      always @(posedge clock)
  // VERILOG-NEXT:   r[2'h1] <= value;
}

// CHECK-LABEL: hw.module private @ConnectSubindexSingleton
hw.module private @ConnectSubindexSingleton(%clock: i1, %reset: i1, %value: i2) -> () {
  %none = hw.constant 0 : i0
  %r = sv.reg : !hw.inout<!hw.array<1xi2>>
  %val = sv.read_inout %r : !hw.inout<!hw.array<1xi2>>
  sv.always posedge %clock {
    sv.passign %r, %1 : !hw.array<1xi2>
  }
  %0 = hw.array_get %val[%none] : !hw.array<1xi2>, i0
  %1 = hw.array_create %value : i2

  // VERILOG:      always @(posedge clock)
  // VERILOG-NEXT:   r[1'h0] <= value;
}

// CHECK-LABEL: hw.module private @ConnectSubindexLeft
hw.module private @ConnectSubindexLeft(%clock: i1, %reset: i1, %value: i2) -> () {
  %c0_i2 = hw.constant 0 : i2

  %r = sv.reg : !hw.inout<!hw.array<3xi2>>
  %val = sv.read_inout %r : !hw.inout<!hw.array<3xi2>>
  sv.always posedge %clock {
    sv.passign %r, %2 : !hw.array<3xi2>
  }

  %0 = hw.array_slice %val[%c0_i2] : (!hw.array<3xi2>) -> !hw.array<2xi2>
  %1 = hw.array_create %value : i2
  %2 = hw.array_concat %1, %0 : !hw.array<1xi2>, !hw.array<2xi2>
  // CHECK: sv.always posedge %clock {
  // CHECK:   [[FIELD_1:%.+]] = sv.array_index_inout %r[%c-2_i2] : !hw.inout<array<3xi2>>, i2
  // CHECK:   sv.passign [[FIELD_1]], %value : i2
  // CHECK: }

  // VERILOG:      always @(posedge clock)
  // VERILOG-NEXT:   r[2'h2] <= value;
}

// CHECK-LABEL: hw.module private @ConnectSubindexRight
hw.module private @ConnectSubindexRight(%clock: i1, %reset: i1, %value: i2) -> () {
  %c1_i2 = hw.constant 1 : i2
  %r = sv.reg : !hw.inout<!hw.array<3xi2>>
  %val = sv.read_inout %r : !hw.inout<!hw.array<3xi2>>
  sv.always posedge %clock {
    sv.passign %r, %2 : !hw.array<3xi2>
  }

  %0 = hw.array_create %value : i2
  %1 = hw.array_slice %val[%c1_i2] : (!hw.array<3xi2>) -> !hw.array<2xi2>
  %2 = hw.array_concat %1, %0 : !hw.array<2xi2>, !hw.array<1xi2>
  // CHECK: sv.always posedge %clock {
  // CHECK:   [[FIELD_0:%.+]] = sv.array_index_inout %r[%c0_i2] : !hw.inout<array<3xi2>>, i2
  // CHECK:   sv.passign [[FIELD_0]], %value : i2
  // CHECK: }

  // VERILOG:      always @(posedge clock)
  // VERILOG-NEXT:     r[2'h0] <= value;
}

// CHECK-LABEL: hw.module private @ConnectSubindices
hw.module private @ConnectSubindices(%clock: i1, %reset: i1, %value: i2) -> () {
  %c0_i3 = hw.constant 0 : i3
  %c2_i3 = hw.constant 2 : i3
  %c3_i3 = hw.constant 3 : i3

  %r = sv.reg : !hw.inout<!hw.array<5xi2>>
  %val = sv.read_inout %r : !hw.inout<!hw.array<5xi2>>
  sv.always posedge %clock {
    sv.passign %r, %8 : !hw.array<5xi2>
  }

  %0 = hw.array_slice %val[%c0_i3] : (!hw.array<5xi2>) -> !hw.array<1xi2>
  %1 = hw.array_create %value : i2
  %2 = hw.array_slice %val[%c2_i3] : (!hw.array<5xi2>) -> !hw.array<3xi2>
  %3 = hw.array_concat %2, %1, %0 : !hw.array<3xi2>, !hw.array<1xi2>, !hw.array<1xi2>
  %4 = hw.array_slice %3[%c0_i3] : (!hw.array<5xi2>) -> !hw.array<2xi2>
  %5 = hw.array_slice %3[%c3_i3] : (!hw.array<5xi2>) -> !hw.array<2xi2>
  %6 = hw.array_concat %5, %1, %4 : !hw.array<2xi2>, !hw.array<1xi2>, !hw.array<2xi2>
  %7 = hw.array_slice %6[%c0_i3] : (!hw.array<5xi2>) -> !hw.array<4xi2>
  %8 = hw.array_concat %1, %7 : !hw.array<1xi2>, !hw.array<4xi2>
  // CHECK: sv.always posedge %clock {
  // CHECK:   [[IDX_1:%.+]] = sv.array_index_inout %r[%c1_i3] : !hw.inout<array<5xi2>>, i3
  // CHECK:   sv.passign [[IDX_1]], %value : i2
  // CHECK:   [[IDX_2:%.+]] = sv.array_index_inout %r[%c2_i3] : !hw.inout<array<5xi2>>, i3
  // CHECK:   sv.passign [[IDX_2:%.+]], %value : i2
  // CHECK:   [[IDX_4:%.+]] = sv.array_index_inout %r[%c-4_i3] : !hw.inout<array<5xi2>>, i3
  // CHECK:   sv.passign [[IDX_4]], %value : i2
  // CHECK: }

  // VERILOG:      always @(posedge clock) begin
  // VERILOG-NEXT:   r[3'h1] <= value;
  // VERILOG-NEXT:   r[3'h2] <= value;
  // VERILOG-NEXT:   r[3'h4] <= value;
  // VERILOG-NEXT: end
}

// CHECK-LABEL: hw.module private @ConnectSubindicesOverwrite
hw.module private @ConnectSubindicesOverwrite(%clock: i1, %reset: i1, %value: i2, %value2: i2) -> () {
  %c0_i3 = hw.constant 0 : i3
  %c2_i3 = hw.constant 2 : i3
  %r = sv.reg : !hw.inout<!hw.array<5xi2>>
  %val = sv.read_inout %r : !hw.inout<!hw.array<5xi2>>
  sv.always posedge %clock {
    sv.passign %r,  %10 : !hw.array<5xi2>
  }

  %0 = hw.array_slice %val[%c0_i3] : (!hw.array<5xi2>) -> !hw.array<1xi2>
  %1 = hw.array_create %value : i2
  %2 = hw.array_slice %val[%c2_i3] : (!hw.array<5xi2>) -> !hw.array<3xi2>
  %3 = hw.array_concat %2, %1, %0 : !hw.array<3xi2>, !hw.array<1xi2>, !hw.array<1xi2>
  %4 = hw.array_slice %3[%c0_i3] : (!hw.array<5xi2>) -> !hw.array<1xi2>
  %5 = hw.array_slice %3[%c2_i3] : (!hw.array<5xi2>) -> !hw.array<3xi2>
  %6 = hw.array_concat %5, %1, %4: !hw.array<3xi2>, !hw.array<1xi2>, !hw.array<1xi2>
  %7 = hw.array_slice %6[%c0_i3] : (!hw.array<5xi2>) -> !hw.array<1xi2>
  %8 = hw.array_create %value2 : i2
  %9 = hw.array_slice %6[%c2_i3] : (!hw.array<5xi2>) -> !hw.array<3xi2>
  %10 = hw.array_concat %9, %8, %7 : !hw.array<3xi2>, !hw.array<1xi2>, !hw.array<1xi2>
  // CHECK: sv.always posedge %clock {
  // CHECK:   [[IDX:%.+]] = sv.array_index_inout %r[%c1_i3] : !hw.inout<array<5xi2>>, i3
  // CHECK:   sv.passign [[IDX]], %value2 : i2
  // CHECK: }


  // VERILOG:      always @(posedge clock)
  // VERILOG-NEXT:   r[3'h1] <= value2;
}

// CHECK-LABEL: hw.module private @ConnectNestedSubindex
hw.module private @ConnectNestedSubindex(%clock: i1, %reset: i1, %value: i2) -> () {
  %c1_i2 = hw.constant 1 : i2
  %c0_i2 = hw.constant 0 : i2
  %c-2_i2 = hw.constant -2 : i2

  %r = sv.reg : !hw.inout<!hw.array<3xarray<3xi2>>>
  %val = sv.read_inout %r : !hw.inout<!hw.array<3xarray<3xi2>>>
  sv.always posedge %clock {
    sv.passign %r, %8 : !hw.array<3xarray<3xi2>>
  }
  %0 = hw.array_get %val[%c1_i2] : !hw.array<3xarray<3xi2>>, i2
  %1 = hw.array_slice %val[%c0_i2] : (!hw.array<3xarray<3xi2>>) -> !hw.array<1xarray<3xi2>>
  %2 = hw.array_slice %0[%c0_i2] : (!hw.array<3xi2>) -> !hw.array<1xi2>
  %3 = hw.array_create %value : i2
  %4 = hw.array_slice %0[%c-2_i2] : (!hw.array<3xi2>) -> !hw.array<1xi2>
  %5 = hw.array_concat %4, %3, %2 : !hw.array<1xi2>, !hw.array<1xi2>, !hw.array<1xi2>
  %6 = hw.array_create %5 : !hw.array<3xi2>
  %7 = hw.array_slice %val[%c-2_i2] : (!hw.array<3xarray<3xi2>>) -> !hw.array<1xarray<3xi2>>
  %8 = hw.array_concat %7, %6, %1 : !hw.array<1xarray<3xi2>>, !hw.array<1xarray<3xi2>>, !hw.array<1xarray<3xi2>>
  // CHECK: sv.always posedge %clock {
  // CHECK:   [[FIELD_INNER:%.+]] = sv.array_index_inout %r[%c1_i2] : !hw.inout<array<3xarray<3xi2>>>, i2
  // CHECK:   [[FIELD_OUTER:%.+]] = sv.array_index_inout [[FIELD_INNER]][%c1_i2_0] : !hw.inout<array<3xi2>>, i2
  // CHECK:   sv.passign [[FIELD_OUTER:%.+]], %value : i2
  // CHECK: }

  // VERILOG:      always @(posedge clock)
  // VERILOG-NEXT:   r[2'h1][2'h1] <= value;
}

// CHECK-LABEL: hw.module private @ConnectNestedFieldsAndIndices
hw.module private @ConnectNestedFieldsAndIndices(%clock: i1, %reset: i1, %value: i2) -> () {
  %c2_i3 = hw.constant 2 : i3
  %c0_i3 = hw.constant 0 : i3
  %c-2_i2 = hw.constant -2 : i2
  %c0_i2 = hw.constant 0 : i2
  %c1_i2 = hw.constant 1 : i2
  %c1_i3 = hw.constant 1 : i3

  %r = sv.reg : !hw.inout<!hw.array<5xstruct<a: !hw.array<3xstruct<b: i2>>>>>
  %val = sv.read_inout %r : !hw.inout<!hw.array<5xstruct<a: !hw.array<3xstruct<b: i2>>>>>
  sv.always posedge %clock {
    sv.passign %r,  %17 : !hw.array<5xstruct<a: !hw.array<3xstruct<b: i2>>>>
  }

  %c1_i3_0 = hw.constant 1 : i3
  %5 = hw.array_get %val[%c1_i3_0] : !hw.array<5xstruct<a: !hw.array<3xstruct<b: i2>>>>, i3
  %6 = hw.struct_extract %5["a"] : !hw.struct<a: !hw.array<3xstruct<b: i2>>>
  %c1_i2_1 = hw.constant 1 : i2
  %7 = hw.array_get %6[%c1_i2_1] : !hw.array<3xstruct<b: i2>>, i2
  %8 = hw.struct_inject %7["b"], %value : !hw.struct<b: i2>
  %9 = hw.array_slice %6[%c0_i2] : (!hw.array<3xstruct<b: i2>>) -> !hw.array<1xstruct<b: i2>>
  %10 = hw.array_create %8 : !hw.struct<b: i2>
  %11 = hw.array_slice %6[%c-2_i2] : (!hw.array<3xstruct<b: i2>>) -> !hw.array<1xstruct<b: i2>>
  %12 = hw.array_concat %11, %10, %9 : !hw.array<1xstruct<b: i2>>, !hw.array<1xstruct<b: i2>>, !hw.array<1xstruct<b: i2>>
  %13 = hw.struct_inject %5["a"], %12 : !hw.struct<a: !hw.array<3xstruct<b: i2>>>
  %14 = hw.array_slice %val[%c0_i3] : (!hw.array<5xstruct<a: !hw.array<3xstruct<b: i2>>>>) -> !hw.array<1xstruct<a: !hw.array<3xstruct<b: i2>>>>
  %15 = hw.array_create %13 : !hw.struct<a: !hw.array<3xstruct<b: i2>>>
  %16 = hw.array_slice %val[%c2_i3] : (!hw.array<5xstruct<a: !hw.array<3xstruct<b: i2>>>>) -> !hw.array<3xstruct<a: !hw.array<3xstruct<b: i2>>>>
  %17 = hw.array_concat %16, %15, %14 : !hw.array<3xstruct<a: !hw.array<3xstruct<b: i2>>>>, !hw.array<1xstruct<a: !hw.array<3xstruct<b: i2>>>>, !hw.array<1xstruct<a: !hw.array<3xstruct<b: i2>>>>

  // CHECK: sv.always posedge %clock {
  // CHECK:   [[ARR_1:%.+]] = sv.array_index_inout %r[%c1_i3] : !hw.inout<array<5xstruct<a: !hw.array<3xstruct<b: i2>>>>>, i3
  // CHECK:   [[STRUCT_A:%.+]] = sv.struct_field_inout [[ARR_1]]["a"] : !hw.inout<struct<a: !hw.array<3xstruct<b: i2>>>>
  // CHECK:   [[ARR_1:%.+]] = sv.array_index_inout [[STRUCT_A]][%c1_i2] : !hw.inout<array<3xstruct<b: i2>>>, i2
  // CHECK:   [[STRUCT_B:%.+]] = sv.struct_field_inout [[ARR_1]]["b"] : !hw.inout<struct<b: i2>>
  // CHECK:   sv.passign [[STRUCT_B]], %value : i2
  // CHECK: }

  // VERILOG:      always @(posedge clock)
  // VERILOG-NEXT:   r[3'h1].a[2'h1].b <= value;
}


// CHECK-LABEL: hw.module private @SelfConnect
hw.module private @SelfConnect(%clock: i1, %reset: i1) -> () {
  %r = sv.reg : !hw.inout<i2>
  %val = sv.read_inout %r : !hw.inout<i2>
  sv.always posedge %clock {
    sv.passign %r, %val : i2
  }

  // CHECK: %r = sv.reg  : !hw.inout<i2>
  // CHECK: sv.always posedge %clock {
  // CHECK:   [[READ:%.+]] = sv.read_inout %r : !hw.inout<i2>
  // CHECK:   sv.passign %r, [[READ]] : i2
  // CHECK: }

  //VERILOG: reg [1:0] r;
  //VERILOG: always @(posedge clock)
  //VERILOG:   r <= r;
}

// CHECK-LABEL: Issue4030
hw.module @Issue4030(%a: i1, %clock: i1, %in1: !hw.array<2xi1>) -> (b: !hw.array<5xi1>) {
  %c0_i3 = hw.constant 0 : i3
  %false = hw.constant false
  %0 = hw.array_get %in1[%false] : !hw.array<2xi1>, i1
  %r = sv.reg  : !hw.inout<array<5xi1>>
  %1 = sv.array_index_inout %r[%c0_i3] : !hw.inout<array<5xi1>>, i3
  %2 = sv.read_inout %r : !hw.inout<array<5xi1>>
  sv.always posedge %clock {
    sv.passign %1, %0 : i1
  }
  hw.output %2 : !hw.array<5xi1>
}
