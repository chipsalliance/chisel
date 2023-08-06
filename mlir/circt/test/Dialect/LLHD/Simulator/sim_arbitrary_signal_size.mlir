// REQUIRES: llhd-sim
// RUN: llhd-sim %s -shared-libs=%shlibdir/libcirct-llhd-signals-runtime-wrappers%shlibext | FileCheck %s

// CHECK: 0ps 0d 0e  root/bool  0x01
// CHECK-NEXT: 0ps 0d 0e  root/fair  0xff00
// CHECK-NEXT: 0ps 0d 0e  root/ginormous 0x000000000000000000000008727f6369aaf83ca15026747af8c7f196ce3f0ad2

llhd.entity @root () -> () {
  %small = hw.constant 1 : i1
  %r = hw.constant 0xff00 : i16
  %b = hw.constant 12345678901234567890123456789012345678901234567890 : i256
  %1 = llhd.sig "bool" %small : i1
  %2 = llhd.sig "fair" %r : i16
  %3 = llhd.sig "ginormous" %b : i256
}
