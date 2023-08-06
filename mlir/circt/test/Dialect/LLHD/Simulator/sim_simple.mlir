// REQUIRES: llhd-sim
// RUN: llhd-sim %s -n 10 -r Foo -shared-libs=%shlibdir/libcirct-llhd-signals-runtime-wrappers%shlibext | FileCheck %s

// CHECK: 0ps 0d 0e  Foo/toggle  0x00
// CHECK-NEXT: 1000ps 0d 0e  Foo/toggle  0x01
// CHECK-NEXT: 2000ps 0d 0e  Foo/toggle  0x00
// CHECK-NEXT: 3000ps 0d 0e  Foo/toggle  0x01
// CHECK-NEXT: 4000ps 0d 0e  Foo/toggle  0x00
// CHECK-NEXT: 5000ps 0d 0e  Foo/toggle  0x01
// CHECK-NEXT: 6000ps 0d 0e  Foo/toggle  0x00
// CHECK-NEXT: 7000ps 0d 0e  Foo/toggle  0x01
// CHECK-NEXT: 8000ps 0d 0e  Foo/toggle  0x00
// CHECK-NEXT: 9000ps 0d 0e  Foo/toggle  0x01
llhd.entity @Foo () -> () {
  %0 = hw.constant 0 : i1
  %toggle = llhd.sig "toggle" %0 : i1
  %1 = llhd.prb %toggle : !llhd.sig<i1>
  %allset = hw.constant 1 : i1
  %2 = comb.xor %1, %allset : i1
  %dt = llhd.constant_time #llhd.time<1ns, 0d, 0e>
  llhd.drv %toggle, %2 after %dt : !llhd.sig<i1>
}
