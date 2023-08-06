// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

// CHECK-LABEL: hw.module @test1(%arg0: i3, %arg1: i1, %arg2: !hw.array<1000xi8>) -> (result: i50) {
hw.module @test1(%arg0: i3, %arg1: i1, %arg2: !hw.array<1000xi8>) -> (result: i50) {
  // CHECK-NEXT:    %c42_i12 = hw.constant 42 : i12
  // CHECK-NEXT:    [[RES0:%[0-9]+]] = comb.add %c42_i12, %c42_i12 : i12
  // CHECK-NEXT:    [[RES1:%[0-9]+]] = comb.mul %c42_i12, [[RES0]] : i12
  %a = hw.constant 42 : i12
  %b = comb.add %a, %a : i12
  %c = comb.mul %a, %b : i12

  // CHECK-NEXT:    [[RES2:%[0-9]+]] = comb.concat %arg0, %arg0, %arg1
  %d = comb.concat %arg0, %arg0, %arg1 : i3, i3, i1

  // CHECK-NEXT:    [[RES4:%[0-9]+]] = comb.concat %c42_i12 : i12
  %conc1 = comb.concat %a : i12

  // CHECK-NEXT:    [[RES7:%[0-9]+]] = comb.parity [[RES4]] : i12
  %parity1 = comb.parity %conc1 : i12

  // CHECK-NEXT:    [[RES8:%[0-9]+]] = comb.concat [[RES4]], [[RES0]], [[RES1]], [[RES2]], [[RES2]] : i12, i12, i12, i7, i7
  %result = comb.concat %conc1, %b, %c, %d, %d : i12, i12, i12, i7, i7

  // CHECK-NEXT: [[RES9:%[0-9]+]] = comb.extract [[RES8]] from 4 : (i50) -> i19
  %small1 = comb.extract %result from 4 : (i50) -> i19

  // CHECK-NEXT: [[RES10:%[0-9]+]] = comb.extract [[RES8]] from 31 : (i50) -> i19
  %small2 = comb.extract %result from 31 : (i50) -> i19

  // CHECK-NEXT: comb.add [[RES9]], [[RES10]] : i19
  %add = comb.add %small1, %small2 : i19

  // CHECK-NEXT: comb.icmp eq [[RES9]], [[RES10]] : i19
  %eq = comb.icmp eq %small1, %small2 : i19

  // CHECK-NEXT: comb.icmp ne [[RES9]], [[RES10]] : i19
  %neq = comb.icmp ne %small1, %small2 : i19

  // CHECK-NEXT: comb.icmp slt [[RES9]], [[RES10]] : i19
  %lt = comb.icmp slt %small1, %small2 : i19

  // CHECK-NEXT: comb.icmp ult [[RES9]], [[RES10]] : i19
  %ult = comb.icmp ult %small1, %small2 : i19

  // CHECK-NEXT: comb.icmp sle [[RES9]], [[RES10]] : i19
  %leq = comb.icmp sle %small1, %small2 : i19

  // CHECK-NEXT: comb.icmp ule [[RES9]], [[RES10]] : i19
  %uleq = comb.icmp ule %small1, %small2 : i19

  // CHECK-NEXT: comb.icmp sgt [[RES9]], [[RES10]] : i19
  %gt = comb.icmp sgt %small1, %small2 : i19

  // CHECK-NEXT: comb.icmp ugt [[RES9]], [[RES10]] : i19
  %ugt = comb.icmp ugt %small1, %small2 : i19

  // CHECK-NEXT: comb.icmp sge [[RES9]], [[RES10]] : i19
  %geq = comb.icmp sge %small1, %small2 : i19

  // CHECK-NEXT: comb.icmp uge [[RES9]], [[RES10]] : i19
  %ugeq = comb.icmp uge %small1, %small2 : i19

  // CHECK-NEXT: %w = sv.wire : !hw.inout<i4>
  %w = sv.wire : !hw.inout<i4>

  // CHECK-NEXT: %after1 = sv.wire : !hw.inout<i4>
  %before1 = sv.wire name "after1" : !hw.inout<i4>

  // CHECK-NEXT: sv.read_inout %after1 : !hw.inout<i4>
  %read_before1 = sv.read_inout %before1 : !hw.inout<i4>

  // CHECK-NEXT: %after2_conflict = sv.wire : !hw.inout<i4>
  // CHECK-NEXT: %after2_conflict_0 = sv.wire name "after2_conflict" : !hw.inout<i4>
  %before2_0 = sv.wire name "after2_conflict" : !hw.inout<i4>
  %before2_1 = sv.wire name "after2_conflict" : !hw.inout<i4>

  // CHECK-NEXT: %after3 = sv.wire {someAttr = "foo"} : !hw.inout<i4>
  %before3 = sv.wire name "after3" {someAttr = "foo"} : !hw.inout<i4>

  // CHECK-NEXT: %w2 = hw.wire [[RES2]] : i7
  %w2 = hw.wire %d : i7

  // CHECK-NEXT: %after4 = hw.wire [[RES2]] : i7
  %before4 = hw.wire %d name "after4" : i7

  // CHECK-NEXT: %after5_conflict = hw.wire [[RES2]] : i7
  // CHECK-NEXT: %after5_conflict_1 = hw.wire [[RES2]] name "after5_conflict" : i7
  %before5_0 = hw.wire %d name "after5_conflict" : i7
  %before5_1 = hw.wire %d name "after5_conflict" : i7

  // CHECK-NEXT: %after6 = hw.wire [[RES2]] {someAttr = "foo"} : i7
  %before6 = hw.wire %d name "after6" {someAttr = "foo"} : i7

  // CHECK-NEXT: = comb.mux %arg1, [[RES2]], [[RES2]] : i7
  %mux = comb.mux %arg1, %d, %d : i7

  // CHECK-NEXT: [[STR:%[0-9]+]] = hw.struct_create ({{.*}}, {{.*}}) : !hw.struct<foo: i19, bar: i7>
  %s0 = hw.struct_create (%small1, %mux) : !hw.struct<foo: i19, bar: i7>

  // CHECK-NEXT: %foo = hw.struct_extract [[STR]]["foo"] : !hw.struct<foo: i19, bar: i7>
  %foo = hw.struct_extract %s0["foo"] : !hw.struct<foo: i19, bar: i7>

  // CHECK-NEXT: = hw.struct_inject [[STR]]["foo"], {{.*}} : !hw.struct<foo: i19, bar: i7>
  %s1 = hw.struct_inject %s0["foo"], %foo : !hw.struct<foo: i19, bar: i7>

  // CHECK-NEXT:  %foo_2, %bar = hw.struct_explode [[STR]] : !hw.struct<foo: i19, bar: i7>
  %foo_2, %bar = hw.struct_explode %s0 : !hw.struct<foo: i19, bar: i7>

  // CHECK-NEXT: hw.bitcast [[STR]] : (!hw.struct<foo: i19, bar: i7>)
  %structBits = hw.bitcast %s0 : (!hw.struct<foo: i19, bar: i7>) -> i26

  // CHECK-NEXT: = arith.constant 13 : i10
  %idx = arith.constant 13 : i10
  // CHECK-NEXT: = hw.array_slice %arg2[%c13_i10] : (!hw.array<1000xi8>) -> !hw.array<24xi8>
  %subArray = hw.array_slice %arg2[%idx] : (!hw.array<1000xi8>) -> !hw.array<24xi8>
  // CHECK-NEXT: [[ARR1:%.+]] = hw.array_create [[RES9]], [[RES10]] : i19
  %arrCreated = hw.array_create %small1, %small2 : i19
  // CHECK-NEXT: [[ARR2:%.+]] = hw.array_create [[RES9]], [[RES10]], {{.+}} : i19
  %arr2 = hw.array_create %small1, %small2, %add : i19
  // CHECK-NEXT: = hw.array_concat [[ARR1]], [[ARR2]] : !hw.array<2xi19>, !hw.array<3xi19>
  %bigArray = hw.array_concat %arrCreated, %arr2 : !hw.array<2 x i19>, !hw.array<3 x i19>
  // CHECK-NEXT: %A = hw.enum.constant A : !hw.enum<A, B, C>
  %A_enum = hw.enum.constant A : !hw.enum<A, B, C>
  // CHECK-NEXT: %B = hw.enum.constant B : !hw.enum<A, B, C>
  %B_enum = hw.enum.constant B : !hw.enum<A, B, C>
  // CHECK-NEXT: = hw.enum.cmp %A, %B : !hw.enum<A, B, C>, !hw.enum<A, B, C>
  %enumcmp = hw.enum.cmp %A_enum, %B_enum : !hw.enum<A, B, C>, !hw.enum<A, B, C>

  // CHECK-NEXT: hw.aggregate_constant [false, true] : !hw.struct<a: i1, b: i1>
  hw.aggregate_constant [false, true] : !hw.struct<a: i1, b: i1>
  //hw.enum.constant A : !hw.enum<A, B>
  // CHECK-NEXT: hw.aggregate_constant [0 : i2, 1 : i2, -2 : i2, -1 : i2] : !hw.array<4xi2>
  hw.aggregate_constant [0 : i2, 1 : i2, -2 : i2, -1 : i2] : !hw.array<4xi2>
  // CHECK-NEXT: hw.aggregate_constant [false] : !hw.uarray<1xi1>
  hw.aggregate_constant [false] : !hw.uarray<1xi1>
  // CHECK-NEXT{LITERAL}: hw.aggregate_constant [[false]] : !hw.struct<a: !hw.array<1xi1>>
  hw.aggregate_constant [[false]] : !hw.struct<a: !hw.array<1xi1>>
  // CHECK-NEXT: hw.aggregate_constant ["A"] : !hw.struct<a: !hw.enum<A, B, C>>
  hw.aggregate_constant ["A"] : !hw.struct<a: !hw.enum<A, B, C>>
  // CHECK-NEXT: hw.aggregate_constant ["A"] : !hw.array<1xenum<A, B, C>>
  hw.aggregate_constant ["A"] : !hw.array<1 x!hw.enum<A, B, C>>

  // CHECK-NEXT:    hw.output [[RES8]] : i50
  hw.output %result : i50
}
// CHECK-NEXT:  }

hw.module @UnionOps(%a: !hw.union<foo: i1, bar: i3>) -> (x: i3, z: !hw.union<bar: i3, baz: i8>) {
  %x = hw.union_extract %a["bar"] : !hw.union<foo: i1, bar: i3>
  %z = hw.union_create "bar", %x : !hw.union<bar: i3, baz: i8>
  hw.output %x, %z : i3, !hw.union<bar: i3, baz: i8>
}
// CHECK-LABEL: hw.module @UnionOps(%a: !hw.union<foo: i1, bar: i3>) -> (x: i3, z: !hw.union<bar: i3, baz: i8>) {
// CHECK-NEXT:    [[I3REG:%.+]] = hw.union_extract %a["bar"] : !hw.union<foo: i1, bar: i3>
// CHECK-NEXT:    [[UREG:%.+]] = hw.union_create "bar", [[I3REG]] : !hw.union<bar: i3, baz: i8>
// CHECK-NEXT:    hw.output [[I3REG]], [[UREG]] : i3, !hw.union<bar: i3, baz: i8>

// https://github.com/llvm/circt/issues/863
// CHECK-LABEL: hw.module @signed_arrays
hw.module @signed_arrays(%arg0: si8) -> (out: !hw.array<2xsi8>) {
  // CHECK-NEXT:  %wireArray = sv.wire  : !hw.inout<array<2xsi8>>
  %wireArray = sv.wire : !hw.inout<!hw.array<2xsi8>>

  // CHECK-NEXT: %0 = hw.array_create %arg0, %arg0 : si8
  %0 = hw.array_create %arg0, %arg0 : si8

  // CHECK-NEXT: sv.assign %wireArray, %0 : !hw.array<2xsi8>
  sv.assign %wireArray, %0 : !hw.array<2xsi8>

  %result = sv.read_inout %wireArray : !hw.inout<!hw.array<2xsi8>>
  hw.output %result : !hw.array<2xsi8>
}

// Check that we pass the verifier that the module's function type matches
// the block argument types when using InOutTypes.
// CHECK: hw.module @InOutPort(%arg0: !hw.inout<i1>)
hw.module @InOutPort(%arg0: !hw.inout<i1>) -> () { }

/// Port names that aren't valid MLIR identifiers are handled with `argNames`
/// attribute being explicitly printed.
// https://github.com/llvm/circt/issues/1822

// CHECK-LABEL: hw.module @argRenames
// CHECK-SAME: attributes {argNames = [""]}
hw.module @argRenames(%arg1: i32) attributes {argNames = [""]} {
}

hw.module @fileListTest(%arg1: i32) attributes {output_filelist = #hw.output_filelist<"foo.f">} {
}

// CHECK-LABEL: hw.module @commentModule
// CHECK-SAME: attributes {comment = "hello world"}
hw.module @commentModule() attributes {comment = "hello world"} {}

module {
// CHECK-LABEL: module {
  hw.globalRef @glbl_B_M1 [#hw.innerNameRef<@A::@inst_1>, #hw.innerNameRef<@B::@memInst>]
  hw.globalRef @glbl_D_M1 [#hw.innerNameRef<@A::@inst_0>, #hw.innerNameRef<@C::@inst>, #hw.innerNameRef<@D::@memInst>]
  hw.globalRef @glbl_D_M2 [#hw.innerNameRef<@A::@inst_0>, #hw.innerNameRef<@C::@inst>, #hw.innerNameRef<@D::@SF>, #hw.innerNameRef<@F::@symA>]
  hw.globalRef @glbl_D_M3 [#hw.innerNameRef<@A::@inst_0>, #hw.innerNameRef<@C::@inst>,   #hw.innerNameRef<@D::@SF>, #hw.innerNameRef<@F::@symB>]

  // CHECK:  hw.globalRef @glbl_B_M1 [#hw.innerNameRef<@A::@inst_1>, #hw.innerNameRef<@B::@memInst>]
  // CHECK:  hw.globalRef @glbl_D_M1 [#hw.innerNameRef<@A::@inst_0>, #hw.innerNameRef<@C::@inst>, #hw.innerNameRef<@D::@memInst>]
  // CHECK:  hw.globalRef @glbl_D_M2 [#hw.innerNameRef<@A::@inst_0>, #hw.innerNameRef<@C::@inst>, #hw.innerNameRef<@D::@SF>, #hw.innerNameRef<@F::@symA>]
  // CHECK:  hw.globalRef @glbl_D_M3 [#hw.innerNameRef<@A::@inst_0>, #hw.innerNameRef<@C::@inst>, #hw.innerNameRef<@D::@SF>, #hw.innerNameRef<@F::@symB>]

  // hw.module.extern @F(%in: i1 {hw.exportPort = #hw<innerSym@symA>}) -> (out: i1 {hw.exportPort = #hw<innerSym@symB>}) attributes {circt.globalRef = [[#hw.globalNameRef<@glbl_D_M2>], [#hw.globalNameRef<@glbl_D_M3>]]}
  hw.module.extern  @F(%in: i1 {hw.exportPort = #hw<innerSym@symA>, circt.globalRef = [#hw.globalNameRef<@glbl_D_M2>]}) -> (out: i1 {hw.exportPort = #hw<innerSym@symB>, circt.globalRef = [#hw.globalNameRef<@glbl_D_M3>]}) attributes {}
  hw.module @F1(%in: i1 {hw.exportPort = #hw<innerSym@symA>, circt.globalRef = [#hw.globalNameRef<@glbl_D_M2>]}) -> (out: i1 {hw.exportPort = #hw<innerSym@symB>, circt.globalRef = [#hw.globalNameRef<@glbl_D_M3>]}) attributes {} {
   hw.output %in : i1
  }
  hw.module @FIRRTLMem() -> () {
  }
  hw.module @D() -> () {
    hw.instance "M1" sym @memInst @FIRRTLMem() -> () {circt.globalRef = [#hw.globalNameRef<@glbl_D_M1>]}
    %c0 = hw.constant 0 : i1
    %2 = hw.instance "ab" sym @SF  @F (in: %c0: i1) -> (out : i1) {circt.globalRef = [#hw.globalNameRef<@glbl_D_M2>, #hw.globalNameRef<@glbl_D_M3>]}
  }
  hw.module @B() -> () {
     hw.instance "M1" sym @memInst @FIRRTLMem() -> () {circt.globalRef = [#hw.globalNameRef<@glbl_B_M1>]}
  }
  hw.module @C() -> () {
    hw.instance "m" sym @inst @D() -> () {circt.globalRef = [#hw.globalNameRef<@glbl_D_M1>, #hw.globalNameRef<@glbl_D_M2>, #hw.globalNameRef<@glbl_D_M3>]}
  }
  hw.module @A() -> () {
    hw.instance "h1" sym @inst_1 @B() -> () {circt.globalRef = [#hw.globalNameRef<@glbl_B_M1>]}
    hw.instance "h2" sym @inst_0 @C() -> () {circt.globalRef = [#hw.globalNameRef<@glbl_D_M1>, #hw.globalNameRef<@glbl_D_M2>, #hw.globalNameRef<@glbl_D_M3>]}
  }
}

module {
  hw.testmodule @NewStyle (input %a : i3, output %b : i3, inout %c : i64 {hw.exportPort = #hw<innerSym@symA>}) {
    hw.output %a : i3
  }
}
