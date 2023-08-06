// REQUIRES: llhd-sim
// RUN: llhd-sim %s -shared-libs=%shlibdir/libcirct-llhd-signals-runtime-wrappers%shlibext | FileCheck %s

// CHECK: 0ps 0d 0e  root/shl  0x01
// CHECK-NEXT: 0ps 0d 0e  root/shrs  0x08
// CHECK-NEXT: 0ps 0d 0e  root/shru  0x08
// CHECK-NEXT: 1000ps 0d 0e  root/shl  0x02
// CHECK-NEXT: 1000ps 0d 0e  root/shrs  0x0c
// CHECK-NEXT: 1000ps 0d 0e  root/shru  0x04
// CHECK-NEXT: 2000ps 0d 0e  root/shl  0x04
// CHECK-NEXT: 2000ps 0d 0e  root/shrs  0x0e
// CHECK-NEXT: 2000ps 0d 0e  root/shru  0x02
// CHECK-NEXT: 3000ps 0d 0e  root/shl  0x08
// CHECK-NEXT: 3000ps 0d 0e  root/shrs  0x0f
// CHECK-NEXT: 3000ps 0d 0e  root/shru  0x01
// CHECK-NEXT: 4000ps 0d 0e  root/shl  0x00
// CHECK-NEXT: 4000ps 0d 0e  root/shru  0x00

llhd.entity @root () -> () {
  %time = llhd.constant_time #llhd.time<1ns, 0d, 0e>

  %init = hw.constant 8 : i4
  %init1 = hw.constant 1 : i4
  %amnt = hw.constant 1 : i4

  %sig = llhd.sig "shrs" %init : i4
  %prbd = llhd.prb %sig : !llhd.sig<i4>
  %shrs = comb.shrs %prbd, %amnt : i4
  llhd.drv %sig, %shrs after %time : !llhd.sig<i4>

  %sig1 = llhd.sig "shru" %init : i4
  %prbd1 = llhd.prb %sig1 : !llhd.sig<i4>
  %shru = comb.shru %prbd1, %amnt : i4
  llhd.drv %sig1, %shru after %time : !llhd.sig<i4>

  %sig2 = llhd.sig "shl" %init1 : i4
  %prbd2 = llhd.prb %sig2 : !llhd.sig<i4>
  %shl = comb.shl %prbd2, %amnt : i4
  llhd.drv %sig2, %shl after %time : !llhd.sig<i4>
}
