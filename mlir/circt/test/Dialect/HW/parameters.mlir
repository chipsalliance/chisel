// RUN: circt-opt %s -verify-diagnostics | circt-opt | FileCheck %s

// CHECK-LABEL: hw.module @parameters<p1: i42 = 17, p2: i1>(%arg0: i8) -> (out: i8) {
hw.module @parameters<p1: i42 = 17, p2: i1>(%arg0: i8) -> (out: i8) {
  hw.output %arg0 : i8
}

// CHECK-LABEL: hw.module @UseParameterized(
hw.module @UseParameterized(%a: i8) -> (xx: i8, yy: i8, zz: i8) {
  // CHECK: %inst1.out = hw.instance "inst1" @parameters<p1: i42 = 4, p2: i1 = false>(arg0:
  %r0 = hw.instance "inst1" @parameters<p1: i42 = 4, p2: i1 = 0>(arg0: %a: i8) -> (out: i8)

  // CHECK: %inst2.out = hw.instance "inst2" @parameters<p1: i42 = 11, p2: i1 = true>(arg0:
  %r1 = hw.instance "inst2" @parameters<p1: i42 = 11, p2: i1 = 1>(arg0: %a: i8) -> (out: i8)

  // CHECK: %inst3.out = hw.instance "inst3" @parameters<p1: i42 = 17, p2: i1 = false>(arg0:
  %r2 = hw.instance "inst3" @parameters<p1: i42 = 17, p2: i1 = 0>(arg0: %a: i8) -> (out: i8)

  hw.output %r0, %r1, %r2: i8, i8, i8
}


// CHECK-LABEL: hw.module.extern @NoArg<param: i42>()
hw.module.extern @NoArg<param: i42>()

// CHECK-LABEL: hw.module @UseParameters<p1: i42>() {
hw.module @UseParameters<p1: i42>() {
  // CHECK: hw.instance "verbatimparam" @NoArg<param: i42 =
  // CHECK-SAME: #hw.param.verbatim<"\22FOO\22">>() -> () 
  hw.instance "verbatimparam" @NoArg<param: i42 = #hw.param.verbatim<"\"FOO\"">>() -> ()

  // CHECK: hw.instance "verbatimparam" @NoArg<param: i42 =
  // CHECK-SAME: #hw.param.expr.add<#hw.param.verbatim<"xxx">, 17>>() -> () 
  hw.instance "verbatimparam" @NoArg<param: i42 = #hw.param.expr.add<#hw.param.verbatim<"xxx">, 17>>() -> () 
  hw.output
}

// CHECK-LABEL: hw.module @affineCanonicalization
hw.module @affineCanonicalization<p1: i4, p2: i4>()
  -> (o: i4, o: i4, o: i4, o: i4, o: i4, o: i4, o: i4, o: i4, o: i4, o: i4, o: i4, o: i4, o: i4, o: i4, o: i4, o: i4) {
  // CHECK-NEXT: %0 = hw.param.value i4 = 6
  %0 = hw.param.value i4 = #hw.param.expr.add<1, 2, 3>
  // CHECK-NEXT: %1 = hw.param.value i4 =
  // CHECK-SAME:     #hw.param.expr.add<#hw.param.expr.mul<#hw.param.decl.ref<"p1">, 2>, 4>
  %1 = hw.param.value i4 = #hw.param.expr.add<#hw.param.decl.ref<"p1">, 4, #hw.param.decl.ref<"p1">>
  // CHECK-NEXT: %2 = hw.param.value i4 =
  // CHECK-SAME:     #hw.param.expr.add<#hw.param.decl.ref<"p1">, 6>
  %2 = hw.param.value i4 = #hw.param.expr.add<2, 4, #hw.param.decl.ref<"p1">>
  // CHECK-NEXT: %3 = hw.param.value i4 =
  // CHECK-SAME:     #hw.param.expr.add<#hw.param.decl.ref<"p1">, #hw.param.decl.ref<"p2">, 4>
  %3 = hw.param.value i4 = #hw.param.expr.add<#hw.param.decl.ref<"p1">, 4, #hw.param.decl.ref<"p2">>

  // CHECK-NEXT: %4 = hw.param.value i4 =
  // CHECK-SAME:     #hw.param.expr.add<#hw.param.expr.mul<#hw.param.decl.ref<"p1">, 2>, 4>
  %4 = hw.param.value i4 = #hw.param.expr.add<#hw.param.expr.add<#hw.param.decl.ref<"p1">, 4>, #hw.param.decl.ref<"p1">>
 
  // CHECK-NEXT: %5 = hw.param.value i4 = #hw.param.decl.ref<"p1">
  %5 = hw.param.value i4 = #hw.param.expr.add<8, #hw.param.decl.ref<"p1">, 8>

  // CHECK-NEXT: %6 = hw.param.value i4 = 0
  %6 = hw.param.value i4 = #hw.param.expr.mul<8, #hw.param.decl.ref<"p1">, 8>

  // CHECK-NEXT: %7 = hw.param.value i4 =
  // CHECK-SAME: #hw.param.expr.mul<#hw.param.decl.ref<"p1">, 4>
  %7 = hw.param.value i4 = #hw.param.expr.shl<#hw.param.decl.ref<"p1">, 2>

  // CHECK-NEXT: %8 = hw.param.value i4 = 
  // CHECK-SAME: #hw.param.expr.add<#hw.param.expr.mul<#hw.param.decl.ref<"p1">, #hw.param.decl.ref<"p2">, 2>,
  // CHECK-SAME:                    #hw.param.expr.mul<#hw.param.decl.ref<"p2">, 6>>
  %8 = hw.param.value i4 = #hw.param.expr.mul<#hw.param.expr.add<#hw.param.decl.ref<"p1">, 3>, 2, #hw.param.decl.ref<"p2">>

  // CHECK-NEXT: %9 = hw.param.value i4 =
  // CHECK-SAME: #hw.param.expr.mul<#hw.param.decl.ref<"p1">, 5>
  %9 = hw.param.value i4 = #hw.param.expr.add<#hw.param.expr.mul<#hw.param.decl.ref<"p1">, 3>, #hw.param.decl.ref<"p1">, #hw.param.decl.ref<"p1">>

  // CHECK-NEXT: %10 = hw.param.value i4 = #hw.param.decl.ref<"p1">
  %10 = hw.param.value i4 = #hw.param.expr.shru<#hw.param.decl.ref<"p1">, 0>

  // CHECK-NEXT: %11 = hw.param.value i4 = #hw.param.decl.ref<"p1">
  %11 = hw.param.value i4 = #hw.param.expr.shrs<#hw.param.decl.ref<"p1">, 0>

  // CHECK-NEXT: %12 = hw.param.value i4 = #hw.param.decl.ref<"p1">
  %12 = hw.param.value i4 = #hw.param.expr.divu<#hw.param.decl.ref<"p1">, 1>

  // CHECK-NEXT: %13 = hw.param.value i4 = #hw.param.decl.ref<"p1">
  %13 = hw.param.value i4 = #hw.param.expr.divs<#hw.param.decl.ref<"p1">, 1>

  // CHECK-NEXT: %14 = hw.param.value i4 = 0
  %14 = hw.param.value i4 = #hw.param.expr.modu<#hw.param.decl.ref<"p1">, 1>

  // CHECK-NEXT: %15 = hw.param.value i4 = 0
  %15 = hw.param.value i4 = #hw.param.expr.mods<#hw.param.decl.ref<"p1">, 1>


  hw.output %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15
     : i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4
}


// CHECK-LABEL: hw.module @associativeOrdering
hw.module @associativeOrdering<p1: i4, p2: i4>()
  -> (o: i4, o: i4, o: i4, o: i4, o: i4, o: i4, o: i4, o: i4) {
  // Declrefs before constants.
  // CHECK-NEXT: %0 = hw.param.value i4 =
  // CHECK-SAME:     #hw.param.expr.add<#hw.param.decl.ref<"p1">, 4>
  %0 = hw.param.value i4 = #hw.param.expr.add<4, #hw.param.decl.ref<"p1">>

  // Declrefs lexically sorted.
  // CHECK-NEXT: %1 = hw.param.value i4 =
  // CHECK-SAME:     #hw.param.expr.add<#hw.param.decl.ref<"p1">, #hw.param.decl.ref<"p2">>
  %1 = hw.param.value i4 =
    #hw.param.expr.add<#hw.param.decl.ref<"p2">, #hw.param.decl.ref<"p1">>

  // Verbatims before declrefs.
  // CHECK-NEXT: %2 = hw.param.value i4 =
  // CHECK-SAME:     #hw.param.expr.add<#hw.param.verbatim<"X">, #hw.param.decl.ref<"p1">>
  %2 = hw.param.value i4 = #hw.param.expr.add<#hw.param.decl.ref<"p1">, #hw.param.verbatim<"X">>

  // Verbatims lexically sorted.
  // CHECK-NEXT: %3 = hw.param.value i4 =
  // CHECK-SAME:     #hw.param.expr.add<#hw.param.verbatim<"X">, #hw.param.verbatim<"Y">>
  %3 = hw.param.value i4 =
    #hw.param.expr.add<#hw.param.verbatim<"Y">, #hw.param.verbatim<"X">>

  // Expressions before Verbatims.

  // CHECK-NEXT: %4 = hw.param.value i4 =
  // CHECK-SAME:     add<#hw.param.expr.mul<{{.*}}>, #hw.param.verbatim<"xxx">>
  %4 = hw.param.value i4 = #hw.param.expr.add<#hw.param.verbatim<"xxx">, #hw.param.expr.mul<#hw.param.decl.ref<"p1">, 4>>
 
  // Expressions sorted by opcode.
  // CHECK-NEXT: %5 = {{.*add<#hw.param.expr.mul<.*>, #hw.param.expr.xor<.*>>}}
  %5 = hw.param.value i4 =
    #hw.param.expr.add<#hw.param.expr.xor<#hw.param.decl.ref<"p1">, 8>,
                       #hw.param.expr.mul<#hw.param.decl.ref<"p1">, 8>>

  // Expression sorted by arity.
  // CHECK-NEXT: %6 = {{.*mul<.*verbatim<"XX">, .*mul<.*decl.ref<"p1">}}
  %6 = hw.param.value i4 =
    #hw.param.expr.add<#hw.param.expr.mul<#hw.param.decl.ref<"p1">, 8>,
                       #hw.param.expr.mul<#hw.param.verbatim<"XX">, #hw.param.verbatim<"YY">, 8>>

  // Expressions sorted by contents.

  // CHECK-NEXT: %7 =
  // CHECK-SAME: #hw.param.expr.mul<{{.*}}verbatim<"XX">
  // CHECK-SAME: #hw.param.expr.mul<{{.*}}decl.ref<"p1">
  %7 = hw.param.value i4 =
    #hw.param.expr.add<#hw.param.expr.mul<#hw.param.decl.ref<"p1">, 8>,
                       #hw.param.expr.mul<#hw.param.verbatim<"XX">, #hw.param.verbatim<"YY">>>

  hw.output %0, %1, %2, %3, %4, %5, %6, %7
     : i4, i4, i4, i4, i4, i4, i4, i4
}

// CHECK-LABEL: hw.module @parameterizedTypes<param: i32>(
hw.module @parameterizedTypes<param: i32>
// CHECK-SAME: %a: i17,
  (%a: !hw.int<17>,
// CHECK-SAME: %b: !hw.int<#hw.param.decl.ref<"param">>) ->
   %b: !hw.int<#hw.param.decl.ref<"param">>) ->
// CHECK-SAME: (c: !hw.int<#hw.param.decl.ref<"param">>)
  (c: !hw.int<#hw.param.decl.ref<"param">>) {

  // CHECK: %paramWire = sv.wire : !hw.inout<int<#hw.param.decl.ref<"param">>>
  %paramWire = sv.wire : !hw.inout<!hw.int<#hw.param.decl.ref<"param">>>
  // CHECK: %0 = sv.read_inout %paramWire : !hw.inout<int<#hw.param.decl.ref<"param">>>
  %0 = sv.read_inout %paramWire : !hw.inout<!hw.int<#hw.param.decl.ref<"param">>>
  // CHECK: hw.output %0 : !hw.int<#hw.param.decl.ref<"param">>
  hw.output %0 : !hw.int<#hw.param.decl.ref<"param">>
}

// CHECK-LABEL: @parameterizedTypesInstance(
hw.module @parameterizedTypesInstance
  (%a: !hw.int<17>, %b: !hw.int<42>) {

  // CHECK: hw.instance "inst" @parameterizedTypes<param: i32 = 42>(a: %a: i17, b: %b: i42) -> (c: i42)
  %c = hw.instance "inst" @parameterizedTypes<param: i32 = 42>
    (a: %a : !hw.int<17>, b: %b : !hw.int<42>) -> (c: !hw.int<42>) {}
}

// CHECK-LABEL: hw.module @parameterizedCombSeq<param: i32>(
hw.module @parameterizedCombSeq<param: i32>
// CHECK-SAME: %a: !hw.int<#hw.param.decl.ref<"param">>
  (%a: !hw.int<#hw.param.decl.ref<"param">>,
    %clk : i1) {

  // CHECK: %0 = comb.add %a, %a : !hw.int<#hw.param.decl.ref<"param">>
  %0 = comb.add %a, %a : !hw.int<#hw.param.decl.ref<"param">>
  // CHECK: %1 = seq.compreg %0, %clk : !hw.int<#hw.param.decl.ref<"param">>
  %1 = seq.compreg %0, %clk: !hw.int<#hw.param.decl.ref<"param">>
}

// CHECK-LABEL: hw.module @CLog2Expression<param: i32>() {
hw.module @CLog2Expression<param: i32>() {
  // CHECK-NEXT: %0 = hw.param.value i32 = 0
  %0 = hw.param.value i32 = #hw.param.expr.clog2<0>

  // CHECK-NEXT: %1 = hw.param.value i32 = 0
  %1 = hw.param.value i32 = #hw.param.expr.clog2<1>

  // CHECK-NEXT: %2 = hw.param.value i32 = 1
  %2 = hw.param.value i32 = #hw.param.expr.clog2<2>

  // CHECK-NEXT: %3 = hw.param.value i32 = 2
  %3 = hw.param.value i32 = #hw.param.expr.clog2<3>

  // CHECK-NEXT: %4 = hw.param.value i32 = #hw.param.expr.clog2<#hw.param.decl.ref<"param">>
  %4 = hw.param.value i32 = #hw.param.expr.clog2<#hw.param.decl.ref<"param">>
}

// CHECK-LABEL: hw.module @parameterizedArrays<param: i32, N: i32>(
hw.module @parameterizedArrays<param: i32, N: i32>
// CHECK-SAME:%a: !hw.array<42xint<#hw.param.decl.ref<"param">>>,
  (%a: !hw.array<42x!hw.int<#hw.param.decl.ref<"param">>>,
// CHECK-SAME: %b: !hw.array<#hw.param.decl.ref<"N">xint<#hw.param.decl.ref<"param">>>) ->
   %b: !hw.array<#hw.param.decl.ref<"N"> x !hw.int<#hw.param.decl.ref<"param">>>) ->
// CHECK-SAME: (c: !hw.array<#hw.param.decl.ref<"N">xint<#hw.param.decl.ref<"param">>>)
   (c: !hw.array<#hw.param.decl.ref<"N"> x !hw.int<#hw.param.decl.ref<"param">>>) {
  hw.output %b : !hw.array<#hw.param.decl.ref<"N"> x !hw.int<#hw.param.decl.ref<"param">>>
}

// CHECK-LABEL: @parameterizedArraysInstance(
hw.module @parameterizedArraysInstance
  (%a: !hw.array<42xint<12>>, %b: !hw.array<24xint<12>>) {

// CHECK:      %inst.c = hw.instance "inst" @parameterizedArrays<param: i32 = 12, N: i32 = 24>
// CHECK-SAME: (a: %a: !hw.array<42xi12>, b: %b: !hw.array<24xi12>) -> (c: !hw.array<24xi12>)
  %c = hw.instance "inst" @parameterizedArrays<param: i32 = 12, N: i32 = 24>
    (a: %a : !hw.array<42xint<12>>, b: %b : !hw.array<24xint<12>>) -> (c: !hw.array<24xint<12>>) {}
}

// CHECK-LABEL: hw.module @NoneTypeParam<p1: none>()
hw.module @NoneTypeParam<p1: none>() -> () {}

// CHECK-LABEL: hw.module @ParamConcatInst() {
// CHECK:         hw.instance "inst" @NoneTypeParam<p1: none = "top.child">() -> ()
hw.module @ParamConcatInst() -> () {
  hw.instance "inst" @NoneTypeParam<p1: none = #hw.param.expr.str.concat<"top", ".", "child">>() -> ()
}
