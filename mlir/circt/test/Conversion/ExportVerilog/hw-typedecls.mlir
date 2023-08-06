// RUN: circt-opt %s -export-verilog -verify-diagnostics | FileCheck %s --strict-whitespace

// CHECK: `ifndef _TYPESCOPE___hw_typedecls
// CHECK: `define _TYPESCOPE___hw_typedecls
hw.type_scope @__hw_typedecls {
  // CHECK: typedef logic foo;
  hw.typedecl @foo : i1
  // CHECK: typedef struct packed {logic a; logic b; } bar;
  hw.typedecl @bar : !hw.struct<a: i1, b: i1>
  hw.typedecl @nest1 : !hw.typealias<@__hw_typedecls::@bar, !hw.struct<a: i1, b: i1>>
  hw.typedecl @nest2 : !hw.typealias<@__hw_typedecls::@nest1, !hw.typealias<@__hw_typedecls::@bar, !hw.struct<a: i1, b: i1>>>

  // CHECK: typedef struct packed {logic a; logic [7:0] b[0:15]; } barArray;
  hw.typedecl @barArray : !hw.struct<a: i1, b: !hw.uarray<16xi8>>

  // CHECK: typedef logic [7:0] baz[0:15];
  hw.typedecl @baz : !hw.uarray<16xi8>
  // CHECK: typedef logic [15:0][7:0] arr;
  hw.typedecl @arr : !hw.array<16xi8>
  // CHECK: typedef logic [31:0] customName;
  hw.typedecl @qux, "customName" : i32
  // CHECK: typedef struct packed {foo a; _other_scope_foo b; } nestedRef;
  hw.typedecl @nestedRef : !hw.struct<a: !hw.typealias<@__hw_typedecls::@foo,i1>, b: !hw.typealias<@_other_scope::@foo,i2>>
  // CHECK: typedef enum bit [1:0] {myEnum_A, myEnum_B, myEnum_C} myEnum;
  hw.typedecl @myEnum : !hw.enum<A, B, C>
}
// CHECK: `endif // _TYPESCOPE___hw_typedecls

// CHECK: `ifndef _TYPESCOPE__other_scope
// CHECK: `define _TYPESCOPE__other_scope
hw.type_scope @_other_scope {
  // CHECK: typedef logic [1:0] _other_scope_foo;
  hw.typedecl @foo, "_other_scope_foo" : i2
}
// CHECK: `endif // _TYPESCOPE__other_scope

// CHECK: `ifndef __PYCDE_TYPES__
// CHECK:   typedef struct packed {logic a; } exTypedef;
// CHECK: `endif
sv.ifdef "__PYCDE_TYPES__"  {
} else  {
  hw.type_scope @pycde  {
    hw.typedecl @exTypedef : !hw.struct<a: i1>
  }
}

// CHECK-LABEL: module testTypeAlias
hw.module @testTypeAlias(
  // CHECK: input  foo      arg0,
  // CHECK:                 arg1
  %arg0: !hw.typealias<@__hw_typedecls::@foo,i1>,
  %arg1: !hw.typealias<@__hw_typedecls::@foo,i1>,
  // CHECK: input  foo[2:0] arg2
  %arg2: !hw.array<3xtypealias<@__hw_typedecls::@foo,i1>>,
  // CHECK: input  arr      arrArg,
  %arrArg: !hw.typealias<@__hw_typedecls::@arr,!hw.array<16xi8>>,
  // CHECK: input  bar      structArg,
  %structArg: !hw.typealias<@__hw_typedecls::@bar,!hw.struct<a: i1, b: i1>>,
  // CHECK: input  myEnum   enumArg,
  %enumArg: !hw.typealias<@__hw_typedecls::@myEnum,!hw.enum<A, B, C>>) ->
  // CHECK: output foo      out
  (out: !hw.typealias<@__hw_typedecls::@foo, i1>) {
  // CHECK: out = arg0 + arg1
  %0 = comb.add %arg0, %arg1 : !hw.typealias<@__hw_typedecls::@foo, i1>
  hw.output %0 : !hw.typealias<@__hw_typedecls::@foo, i1>
}

// CHECK-LABEL: module testRegOp
hw.module @testRegOp() -> () {
  // CHECK: foo {{.+}};
  %r1 = sv.reg : !hw.inout<!hw.typealias<@__hw_typedecls::@foo,i1>>
  // CHECK: foo[2:0] {{.+}};
  %r2 = sv.reg : !hw.inout<!hw.array<3xtypealias<@__hw_typedecls::@foo,i1>>>
}

// CHECK-LABEL: module testAggregateCreate
hw.module @testAggregateCreate(%i: i1) -> (out1: i1, out2: i1) {
  // CHECK: wire bar [[NAME:.+]] = {{.+}};
  %0 = hw.struct_create(%i, %i) : !hw.typealias<@__hw_typedecls::@bar,!hw.struct<a: i1, b: i1>>
  // CHECK: [[NAME]].a
  %1 = hw.struct_extract %0["a"] : !hw.typealias<@__hw_typedecls::@bar,!hw.struct<a: i1, b: i1>>
  // CHECK: [[NAME]].b
  %2 = hw.struct_extract %0["b"] : !hw.typealias<@__hw_typedecls::@bar,!hw.struct<a: i1, b: i1>>
  hw.output %1, %2 : i1, i1
}

hw.module @testNestedAlias(%i: !hw.typealias<@__hw_typedecls::@nest2, !hw.typealias<@__hw_typedecls::@nest1, !hw.typealias<@__hw_typedecls::@bar, !hw.struct<a: i1, b: i1>>>>) -> () {
  %0 = hw.struct_extract %i["a"] : !hw.typealias<@__hw_typedecls::@nest2, !hw.typealias<@__hw_typedecls::@nest1, !hw.typealias<@__hw_typedecls::@bar, !hw.struct<a: i1, b: i1>>>>
}

// CHECK-LABEL: module testAggregateInout
hw.module @testAggregateInout(%i: i1) -> (out1: i8, out2: i1) {
  // CHECK:      wire arr array;
  // CHECK-NEXT: wire bar str;
  // CHECK-NEXT: assign out1 = array[4'h0];
  // CHECK-NEXT: assign out2 = str.a;
  %array = sv.wire : !hw.inout<!hw.typealias<@__hw_typedecls::@arr, !hw.array<16xi8>>>
  %str = sv.wire : !hw.inout<!hw.typealias<@__hw_typedecls::@bar, !hw.struct<a: i1, b: i1>>>
  %c0_i4 = hw.constant 0 : i4
  %0 = sv.array_index_inout %array[%c0_i4] : !hw.inout<!hw.typealias<@__hw_typedecls::@arr, !hw.array<16xi8>>>, i4
  %1 = sv.struct_field_inout %str["a"] : !hw.inout<!hw.typealias<@__hw_typedecls::@bar, !hw.struct<a: i1, b: i1>>>
  %2 = sv.read_inout %0 : !hw.inout<i8>
  %3 = sv.read_inout %1 : !hw.inout<i1>
  hw.output %2, %3 : i8, i1
}

// CHECK-LABEL: module testEnumOps
hw.module @testEnumOps() -> (out1: !hw.typealias<@__hw_typedecls::@myEnum,!hw.enum<A, B, C>>) {
  // CHECK: assign out1 = myEnum_A;
  %0 = hw.enum.constant A : !hw.typealias<@__hw_typedecls::@myEnum,!hw.enum<A, B, C>>
  hw.output %0 : !hw.typealias<@__hw_typedecls::@myEnum,!hw.enum<A, B, C>>
}
