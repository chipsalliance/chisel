// RUN: circt-opt %s | circt-opt -verify-diagnostics | FileCheck %s
// RUN: circt-opt %s -canonicalize | circt-opt -verify-diagnostics | FileCheck %s --check-prefix=CANON
// RUN: circt-opt %s --lower-esi-types -canonicalize | circt-opt -verify-diagnostics | FileCheck %s --check-prefix=LOW


!TypeA = !hw.struct<header1: i6, header2: i1, header3: !hw.array<13xi16>>
!TypeAwin1 = !esi.window<
  "TypeAwin1", !TypeA, [
    <"FrameA", [
      <"header1">,
      <"header2">
    ]>,
    <"FrameB", [
      <"header3", 3>
    ]>
  ]>


// CHECK-LABEL:   hw.module.extern @TypeAModuleDst(%windowed: !esi.window<"TypeAwin1", !hw.struct<header1: i6, header2: i1, header3: !hw.array<13xi16>>, [<"FrameA", [<"header1">, <"header2">]>, <"FrameB", [<"header3", 3>]>]>)
hw.module.extern @TypeAModuleDst(%windowed: !TypeAwin1)
// CHECK-LABEL:   hw.module.extern @TypeAModuleSrc() -> (windowed: !esi.window<"TypeAwin1", !hw.struct<header1: i6, header2: i1, header3: !hw.array<13xi16>>, [<"FrameA", [<"header1">, <"header2">]>, <"FrameB", [<"header3", 3>]>]>)
hw.module.extern @TypeAModuleSrc() -> (windowed: !TypeAwin1)

!lowered = !hw.union<
  FrameA: !hw.struct<header1: i6, header2: i1>,
  FrameB: !hw.struct<header3: !hw.array<3xi16>>,
  FrameB_leftOver: !hw.struct<header3: !hw.array<1xi16>>>

// CHECK-LABEL: hw.module @TypeAModuleUnwrap
// CHECK:         [[r0:%.+]] = esi.window.unwrap %a : !esi.window<"TypeAwin1", !hw.struct<header1: i6, header2: i1, header3: !hw.array<13xi16>>, [<"FrameA", [<"header1">, <"header2">]>, <"FrameB", [<"header3", 3>]>]>
// CHECK:         hw.output [[r0]] : !hw.union<FrameA: !hw.struct<header1: i6, header2: i1>, FrameB: !hw.struct<header3: !hw.array<3xi16>>, FrameB_leftOver: !hw.struct<header3: !hw.array<1xi16>>>
hw.module @TypeAModuleUnwrap(%a: !TypeAwin1) -> (x: !lowered) {
  %u = esi.window.unwrap %a : !TypeAwin1
  hw.output %u : !lowered
}

// CHECK-LABEL: hw.module @TypeAModuleUnwrapWrap
// CHECK:         [[r0:%.+]] = esi.window.unwrap %a : !esi.window<"TypeAwin1", !hw.struct<header1: i6, header2: i1, header3: !hw.array<13xi16>>, [<"FrameA", [<"header1">, <"header2">]>, <"FrameB", [<"header3", 3>]>]>
// CHECK:         [[r1:%.+]] = esi.window.wrap [[r0]] : !esi.window<"TypeAwin1", !hw.struct<header1: i6, header2: i1, header3: !hw.array<13xi16>>, [<"FrameA", [<"header1">, <"header2">]>, <"FrameB", [<"header3", 3>]>]>
// CANON-LABEL: hw.module @TypeAModuleUnwrapWrap
// CANON-NEXT:    hw.output %a
hw.module @TypeAModuleUnwrapWrap(%a: !TypeAwin1) -> (x: !TypeAwin1) {
  %u = esi.window.unwrap %a : !TypeAwin1
  %x = esi.window.wrap %u : !TypeAwin1
  hw.output %x : !TypeAwin1
}

// CANON-LABEL: hw.module @TypeAModuleWrapUnwrap
// CANON-NEXT:    hw.output %a
hw.module @TypeAModuleWrapUnwrap(%a: !lowered) -> (x: !lowered) {
  %w = esi.window.wrap %a : !TypeAwin1
  %u = esi.window.unwrap %w : !TypeAwin1
  hw.output %u : !lowered
}

// LOW-LABEL:  hw.module @TypeAModulePassthrough(%a: !hw.union<FrameA: !hw.struct<header1: i6, header2: i1>, FrameB: !hw.struct<header3: !hw.array<3xi16>>, FrameB_leftOver: !hw.struct<header3: !hw.array<1xi16>>>) -> (x: !hw.union<FrameA: !hw.struct<header1: i6, header2: i1>, FrameB: !hw.struct<header3: !hw.array<3xi16>>, FrameB_leftOver: !hw.struct<header3: !hw.array<1xi16>>>) {
// LOW-NEXT:     %foo.x = hw.instance "foo" @TypeAModuleUnwrapWrap(a: %a: !hw.union<FrameA: !hw.struct<header1: i6, header2: i1>, FrameB: !hw.struct<header3: !hw.array<3xi16>>, FrameB_leftOver: !hw.struct<header3: !hw.array<1xi16>>>) -> (x: !hw.union<FrameA: !hw.struct<header1: i6, header2: i1>, FrameB: !hw.struct<header3: !hw.array<3xi16>>, FrameB_leftOver: !hw.struct<header3: !hw.array<1xi16>>>)
// LOW-NEXT:     hw.output %foo.x :  !hw.union<FrameA: !hw.struct<header1: i6, header2: i1>, FrameB: !hw.struct<header3: !hw.array<3xi16>>, FrameB_leftOver: !hw.struct<header3: !hw.array<1xi16>>>
hw.module @TypeAModulePassthrough(%a: !TypeAwin1) -> (x: !TypeAwin1) {
  %x = hw.instance "foo" @TypeAModuleUnwrapWrap(a: %a: !TypeAwin1) -> (x: !TypeAwin1)
  hw.output %x : !TypeAwin1
}
