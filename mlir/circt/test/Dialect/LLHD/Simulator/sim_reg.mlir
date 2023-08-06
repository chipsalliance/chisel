// REQUIRES: llhd-sim
// RUN: llhd-sim %s -n 10 -shared-libs=%shlibdir/libcirct-llhd-signals-runtime-wrappers%shlibext | FileCheck %s

// CHECK: 0ps 0d 0e  root/clock  0x00
// CHECK-NEXT: 0ps 0d 0e  root/sig1  0x00000000
// CHECK-NEXT: 0ps 0d 0e  root/sig2  0x00000000
// CHECK-NEXT: 1000ps 0d 0e  root/sig1  0x00000002
// CHECK-NEXT: 2000ps 0d 0e  root/clock  0x01
// CHECK-NEXT: 3000ps 0d 0e  root/sig1  0x00000001
// CHECK-NEXT: 3000ps 0d 0e  root/sig2  0xffffffff
// CHECK-NEXT: 4000ps 0d 0e  root/clock  0x00
// CHECK-NEXT: 4000ps 0d 0e  root/sig1  0x00000003
// CHECK-NEXT: 5000ps 0d 0e  root/sig1  0x00000000
// CHECK-NEXT: 5000ps 0d 0e  root/sig2  0x00000000
// CHECK-NEXT: 6000ps 0d 0e  root/clock  0x01
// CHECK-NEXT: 6000ps 0d 0e  root/sig1  0x00000002
// CHECK-NEXT: 7000ps 0d 0e  root/sig1  0x00000001
// CHECK-NEXT: 7000ps 0d 0e  root/sig2  0xffffffff
// CHECK-NEXT: 8000ps 0d 0e  root/clock  0x00
// CHECK-NEXT: 8000ps 0d 0e  root/sig1  0x00000003
// CHECK-NEXT: 9000ps 0d 0e  root/sig1  0x00000000
// CHECK-NEXT: 9000ps 0d 0e  root/sig2  0x00000000
llhd.entity @root () -> () {
  %0 = llhd.constant_time #llhd.time<1ns, 0d, 0e>
  %t1 = llhd.constant_time #llhd.time<2ns, 0d, 0e>
  %1 = hw.constant 0 : i1
  %c0 = hw.constant 0 : i32
  %c1 = hw.constant 1 : i32
  %c2 = hw.constant 2 : i32
  %c3 = hw.constant 3 : i32
  %s0 = llhd.sig "sig1" %c0 : i32
  %s1 = llhd.sig "sig2" %c0 : i32
  %c = llhd.sig "clock" %1 : i1
  %p = llhd.prb %c : !llhd.sig<i1>
  %allset1 = hw.constant 1 : i1
  %nc = comb.xor %p, %allset1 : i1
  llhd.drv %c, %nc after %t1 : !llhd.sig<i1>
  llhd.reg %s0, (%c0, "fall" %p after %0 : i32), (%c1, "rise" %p after %0 : i32), (%c2, "low" %p after %0 : i32), (%c3, "high" %p after %0 : i32) : !llhd.sig<i32>
  %2 = llhd.prb %s1 : !llhd.sig<i32>
  %allset32 = hw.constant -1 : i32
  %3 = comb.xor %2, %allset32 : i32
  llhd.reg %s1, (%3, "both" %p after %0 : i32) : !llhd.sig<i32>
}
