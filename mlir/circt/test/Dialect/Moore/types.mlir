// RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK-LABEL: func @UnitTypes(
func.func @UnitTypes(
  // CHECK-SAME: %arg0: !moore.void
  // CHECK-SAME: %arg1: !moore.string
  // CHECK-SAME: %arg2: !moore.chandle
  // CHECK-SAME: %arg3: !moore.event
  %arg0: !moore.void,
  %arg1: !moore.string,
  %arg2: !moore.chandle,
  %arg3: !moore.event
) { return }

// CHECK-LABEL: func @IntTypes(
func.func @IntTypes(
  // CHECK-SAME: %arg0: !moore.bit
  // CHECK-SAME: %arg1: !moore.logic
  // CHECK-SAME: %arg2: !moore.reg
  // CHECK-SAME: %arg3: !moore.byte
  // CHECK-SAME: %arg4: !moore.shortint
  // CHECK-SAME: %arg5: !moore.int
  // CHECK-SAME: %arg6: !moore.longint
  // CHECK-SAME: %arg7: !moore.integer
  // CHECK-SAME: %arg8: !moore.time
  %arg0: !moore.bit,
  %arg1: !moore.logic,
  %arg2: !moore.reg,
  %arg3: !moore.byte,
  %arg4: !moore.shortint,
  %arg5: !moore.int,
  %arg6: !moore.longint,
  %arg7: !moore.integer,
  %arg8: !moore.time,
  // CHECK-SAME: %arg9: !moore.bit<unsigned>
  // CHECK-SAME: %arg10: !moore.logic<unsigned>
  // CHECK-SAME: %arg11: !moore.reg<unsigned>
  // CHECK-SAME: %arg12: !moore.byte<unsigned>
  // CHECK-SAME: %arg13: !moore.shortint<unsigned>
  // CHECK-SAME: %arg14: !moore.int<unsigned>
  // CHECK-SAME: %arg15: !moore.longint<unsigned>
  // CHECK-SAME: %arg16: !moore.integer<unsigned>
  // CHECK-SAME: %arg17: !moore.time<unsigned>
  %arg9: !moore.bit<unsigned>,
  %arg10: !moore.logic<unsigned>,
  %arg11: !moore.reg<unsigned>,
  %arg12: !moore.byte<unsigned>,
  %arg13: !moore.shortint<unsigned>,
  %arg14: !moore.int<unsigned>,
  %arg15: !moore.longint<unsigned>,
  %arg16: !moore.integer<unsigned>,
  %arg17: !moore.time<unsigned>,
  // CHECK-SAME: %arg18: !moore.bit<signed>
  // CHECK-SAME: %arg19: !moore.logic<signed>
  // CHECK-SAME: %arg20: !moore.reg<signed>
  // CHECK-SAME: %arg21: !moore.byte<signed>
  // CHECK-SAME: %arg22: !moore.shortint<signed>
  // CHECK-SAME: %arg23: !moore.int<signed>
  // CHECK-SAME: %arg24: !moore.longint<signed>
  // CHECK-SAME: %arg25: !moore.integer<signed>
  // CHECK-SAME: %arg26: !moore.time<signed>
  %arg18: !moore.bit<signed>,
  %arg19: !moore.logic<signed>,
  %arg20: !moore.reg<signed>,
  %arg21: !moore.byte<signed>,
  %arg22: !moore.shortint<signed>,
  %arg23: !moore.int<signed>,
  %arg24: !moore.longint<signed>,
  %arg25: !moore.integer<signed>,
  %arg26: !moore.time<signed>
) { return }

// CHECK-LABEL: func @RealTypes(
func.func @RealTypes(
  // CHECK-SAME: %arg0: !moore.shortreal
  // CHECK-SAME: %arg1: !moore.real
  // CHECK-SAME: %arg2: !moore.realtime
  %arg0: !moore.shortreal,
  %arg1: !moore.real,
  %arg2: !moore.realtime
) { return }

// CHECK-LABEL: func @EnumType(
func.func @EnumType(
  // CHECK-SAME: %arg0: !moore.enum<loc("foo.sv":42:9001)>
  // CHECK-SAME: %arg1: !moore.enum<int, loc("foo.sv":42:9001)>
  // CHECK-SAME: %arg2: !moore.enum<"Foo", loc("foo.sv":42:9001)>
  // CHECK-SAME: %arg3: !moore.enum<"Foo", int, loc("foo.sv":42:9001)>
  %arg0: !moore.enum<loc("foo.sv":42:9001)>,
  %arg1: !moore.enum<int, loc("foo.sv":42:9001)>,
  %arg2: !moore.enum<"Foo", loc("foo.sv":42:9001)>,
  %arg3: !moore.enum<"Foo", int, loc("foo.sv":42:9001)>
) { return }

// CHECK-LABEL: func @IndirectTypes(
func.func @IndirectTypes(
  // CHECK-SAME: %arg0: !moore.packed<named<"Foo", bit, loc("foo.sv":42:9001)>>
  // CHECK-SAME: %arg1: !moore.packed<ref<bit, loc("foo.sv":42:9001)>>
  %arg0: !moore.packed<named<"Foo", bit, loc("foo.sv":42:9001)>>,
  %arg1: !moore.packed<ref<bit, loc("foo.sv":42:9001)>>,
  // CHECK-SAME: %arg2: !moore.unpacked<named<"Foo", bit, loc("foo.sv":42:9001)>>
  // CHECK-SAME: %arg3: !moore.unpacked<named<"Foo", string, loc("foo.sv":42:9001)>>
  // CHECK-SAME: %arg4: !moore.unpacked<ref<bit, loc("foo.sv":42:9001)>>
  // CHECK-SAME: %arg5: !moore.unpacked<ref<string, loc("foo.sv":42:9001)>>
  %arg2: !moore.unpacked<named<"Foo", bit, loc("foo.sv":42:9001)>>,
  %arg3: !moore.unpacked<named<"Foo", string, loc("foo.sv":42:9001)>>,
  %arg4: !moore.unpacked<ref<bit, loc("foo.sv":42:9001)>>,
  %arg5: !moore.unpacked<ref<string, loc("foo.sv":42:9001)>>
) { return }

// CHECK-LABEL: func @DimTypes(
func.func @DimTypes(
  // CHECK-SAME: %arg0: !moore.packed<unsized<bit>>,
  // CHECK-SAME: %arg1: !moore.packed<range<bit, 4:-5>>,
  %arg0: !moore.packed<unsized<bit>>,
  %arg1: !moore.packed<range<bit, 4:-5>>,
  // CHECK-SAME: %arg2: !moore.unpacked<unsized<bit>>,
  // CHECK-SAME: %arg3: !moore.unpacked<array<bit, 42>>,
  // CHECK-SAME: %arg4: !moore.unpacked<range<bit, 4:-5>>,
  // CHECK-SAME: %arg5: !moore.unpacked<assoc<bit>>,
  // CHECK-SAME: %arg6: !moore.unpacked<assoc<bit, string>>,
  // CHECK-SAME: %arg7: !moore.unpacked<queue<bit>>,
  // CHECK-SAME: %arg8: !moore.unpacked<queue<bit, 9001>>
  %arg2: !moore.unpacked<unsized<bit>>,
  %arg3: !moore.unpacked<array<bit, 42>>,
  %arg4: !moore.unpacked<range<bit, 4:-5>>,
  %arg5: !moore.unpacked<assoc<bit>>,
  %arg6: !moore.unpacked<assoc<bit, string>>,
  %arg7: !moore.unpacked<queue<bit>>,
  %arg8: !moore.unpacked<queue<bit, 9001>>
) {
  return
}

// CHECK-LABEL: func @StructTypes(
func.func @StructTypes(
  // CHECK-SAME: %arg0: !moore.packed<struct<{}, loc("foo.sv":42:9001)>>
  // CHECK-SAME: %arg1: !moore.packed<struct<"Foo", {}, loc("foo.sv":42:9001)>>
  // CHECK-SAME: %arg2: !moore.packed<struct<unsigned, {}, loc("foo.sv":42:9001)>>
  // CHECK-SAME: %arg3: !moore.packed<struct<"Foo", signed, {}, loc("foo.sv":42:9001)>>
  // CHECK-SAME: %arg4: !moore.packed<struct<{foo: bit loc("foo.sv":1:2), bar: int loc("foo.sv":3:4)}, loc("foo.sv":42:9001)>>
  %arg0: !moore.packed<struct<{}, loc("foo.sv":42:9001)>>,
  %arg1: !moore.packed<struct<"Foo", {}, loc("foo.sv":42:9001)>>,
  %arg2: !moore.packed<struct<unsigned, {}, loc("foo.sv":42:9001)>>,
  %arg3: !moore.packed<struct<"Foo", signed, {}, loc("foo.sv":42:9001)>>,
  %arg4: !moore.packed<struct<{foo: bit loc("foo.sv":1:2), bar: int loc("foo.sv":3:4)}, loc("foo.sv":42:9001)>>,
  // CHECK-SAME: %arg5: !moore.unpacked<struct<{}, loc("foo.sv":42:9001)>>
  // CHECK-SAME: %arg6: !moore.unpacked<struct<"Foo", {}, loc("foo.sv":42:9001)>>
  // CHECK-SAME: %arg7: !moore.unpacked<struct<{foo: string loc("foo.sv":1:2), bar: event loc("foo.sv":3:4)}, loc("foo.sv":42:9001)>>
  %arg5: !moore.unpacked<struct<{}, loc("foo.sv":42:9001)>>,
  %arg6: !moore.unpacked<struct<"Foo", {}, loc("foo.sv":42:9001)>>,
  %arg7: !moore.unpacked<struct<{foo: string loc("foo.sv":1:2), bar: event loc("foo.sv":3:4)}, loc("foo.sv":42:9001)>>
) { return }
