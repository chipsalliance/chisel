// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s
// RUN: circt-opt %s --msft-discover-appids -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s --check-prefix=APPID
// RUN: circt-opt %s --lower-msft-to-hw -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s --check-prefix=HWLOW

// CHECK-LABEL: msft.module @top
// APPID-LABEL: msft.module @top {} () attributes {childAppIDBases = ["extern", "foo"]}
// HWLOW-LABEL: hw.module @top
msft.module @top {} () {
  msft.instance @foo @Foo() {msft.appid = #msft.appid<"foo"[0]>} : () -> (i32)
  // CHECK: %foo.x = msft.instance @foo @Foo() {msft.appid = #msft.appid<"foo"[0]>} : () -> i32
  // HWLOW: %foo.x = hw.instance "foo" sym @foo @Foo<__INST_HIER: none = #hw.param.expr.str.concat<#hw.param.decl.ref<"__INST_HIER">, ".foo">>() -> (x: i32)

  %true = hw.constant true
  %extern.out = msft.instance @extern @Extern(%true)<param: i1 = false> : (i1) -> i1
  // CHECK: %extern.out = msft.instance @extern @Extern(%true) <param: i1 = false> : (i1) -> i1
  // HWLOW: %extern.out = hw.instance "extern" sym @extern @Extern<param: i1 = false>(in: %true: i1) -> (out: i1)
  msft.output
}

// CHECK-LABEL: msft.module @B {WIDTH = 1 : i64} (%a: i4) -> (nameOfPortInSV: i4) attributes {fileName = "b.sv"} {
// HWLOW-LABEL: hw.module @B<__INST_HIER: none = "INSTANTIATE_WITH_INSTANCE_PATH">(%a: i4) -> (nameOfPortInSV: i4) attributes {output_file = #hw.output_file<"b.sv", includeReplicatedOps>} {
msft.module @B { "WIDTH" = 1 } (%a: i4) -> (nameOfPortInSV: i4) attributes {fileName = "b.sv"} {
  %0 = comb.add %a, %a : i4
  // CHECK: comb.add %a, %a : i4
  // HWLOW: comb.add %a, %a : i4
  msft.output %0: i4
}

// CHECK-LABEL: msft.module @UnGenerated {DEPTH = 3 : i64} (%a: i1) -> (nameOfPortInSV: i1)
// HWLOW-LABEL: Module not generated: \22UnGenerated\22 params {DEPTH = 3 : i64}
msft.module @UnGenerated { DEPTH = 3 } (%a: i1) -> (nameOfPortInSV: i1)

// APPID-LABEL: msft.module @Foo {WIDTH = 1 : i64} () -> (x: i32) attributes {childAppIDBases = ["extern", "foo"]}
msft.module @Foo { "WIDTH" = 1 } () -> (x: i32) {
  %true = hw.constant true
  msft.instance @extern @Extern(%true)<param: i1 = false> {msft.appid=#msft.appid<"extern"[0]>} : (i1) -> i1
  %c1_4 = hw.constant 1 : i4
  msft.instance @b @B(%c1_4) {msft.appid=#msft.appid<"foo"[0]>} : (i4) -> i4
  %c0 = hw.constant 0 : i32
  msft.output %c0 : i32
}

// CHECK-LABEL: msft.module.extern @Extern<param: i1>(%in: i1) -> (out: i1)
// HWLOW-LABEL: hw.module.extern @Extern<param: i1>(%in: i1) -> (out: i1)
msft.module.extern @Extern<param: i1> (%in: i1) -> (out: i1)

// HWLOW-LABEL: esi.pure_module @PureMod
esi.pure_module @PureMod {
  // HWLOW-NEXT: hw.instance "top" sym @top @top<__INST_HIER: none = "PureMod.top">() -> ()
  msft.instance @top @top() {} : () -> ()
}
