// REQUIRES: llhd-sim
// RUN: llhd-sim %s -T 5000 --trace-format=full -shared-libs=%shlibdir/libcirct-llhd-signals-runtime-wrappers%shlibext | FileCheck %s --check-prefix=FULL
// RUN: llhd-sim %s -T 5000 --trace-format=reduced -shared-libs=%shlibdir/libcirct-llhd-signals-runtime-wrappers%shlibext | FileCheck %s --check-prefix=REDUCED
// RUN: llhd-sim %s -T 5000 --trace-format=merged -shared-libs=%shlibdir/libcirct-llhd-signals-runtime-wrappers%shlibext | FileCheck %s --check-prefix=MERGED
// RUN: llhd-sim %s -T 5000 --trace-format=merged-reduce -shared-libs=%shlibdir/libcirct-llhd-signals-runtime-wrappers%shlibext | FileCheck %s --check-prefix=MERGEDRED
// RUN: llhd-sim %s -T 5000 --trace-format=named-only -shared-libs=%shlibdir/libcirct-llhd-signals-runtime-wrappers%shlibext | FileCheck %s --check-prefix=NAMED

// FULL: 0ps 0d 0e  root/1  0x01
// FULL: 0ps 0d 0e  root/foo/s  0x01
// FULL: 0ps 0d 0e  root/s  0x01
// FULL: 0ps 0d 1e  root/foo/s  0x02
// FULL: 0ps 0d 1e  root/s  0x02
// FULL: 0ps 0d 2e  root/foo/s  0x03
// FULL: 0ps 0d 2e  root/s  0x03
// FULL: 1000ps 0d 1e  root/foo/s  0x06
// FULL: 1000ps 0d 1e  root/s  0x06
// FULL: 1000ps 0d 2e  root/foo/s  0x09
// FULL: 1000ps 0d 2e  root/s  0x09
// FULL: 2000ps 0d 1e  root/foo/s  0x12
// FULL: 2000ps 0d 1e  root/s  0x12
// FULL: 2000ps 0d 2e  root/foo/s  0x1b
// FULL: 2000ps 0d 2e  root/s  0x1b
// FULL: 3000ps 0d 1e  root/foo/s  0x36
// FULL: 3000ps 0d 1e  root/s  0x36
// FULL: 3000ps 0d 2e  root/foo/s  0x51
// FULL: 3000ps 0d 2e  root/s  0x51
// FULL: 4000ps 0d 1e  root/foo/s  0xa2
// FULL: 4000ps 0d 1e  root/s  0xa2
// FULL: 4000ps 0d 2e  root/foo/s  0xf3
// FULL: 4000ps 0d 2e  root/s  0xf3
// FULL: 5000ps 0d 1e  root/foo/s  0xe6
// FULL: 5000ps 0d 1e  root/s  0xe6
// FULL: 5000ps 0d 2e  root/foo/s  0xd9
// FULL: 5000ps 0d 2e  root/s  0xd9

// REDUCED: 0ps 0d 0e  root/1  0x01
// REDUCED: 0ps 0d 0e  root/s  0x01
// REDUCED: 0ps 0d 1e  root/s  0x02
// REDUCED: 0ps 0d 2e  root/s  0x03
// REDUCED: 1000ps 0d 1e  root/s  0x06
// REDUCED: 1000ps 0d 2e  root/s  0x09
// REDUCED: 2000ps 0d 1e  root/s  0x12
// REDUCED: 2000ps 0d 2e  root/s  0x1b
// REDUCED: 3000ps 0d 1e  root/s  0x36
// REDUCED: 3000ps 0d 2e  root/s  0x51
// REDUCED: 4000ps 0d 1e  root/s  0xa2
// REDUCED: 4000ps 0d 2e  root/s  0xf3
// REDUCED: 5000ps 0d 1e  root/s  0xe6
// REDUCED: 5000ps 0d 2e  root/s  0xd9

// MERGED: 0ps
// MERGED:   root/1  0x01
// MERGED:   root/foo/s  0x03
// MERGED:   root/s  0x03
// MERGED: 1000ps
// MERGED:   root/foo/s  0x09
// MERGED:   root/s  0x09
// MERGED: 2000ps
// MERGED:   root/foo/s  0x1b
// MERGED:   root/s  0x1b
// MERGED: 3000ps
// MERGED:   root/foo/s  0x51
// MERGED:   root/s  0x51
// MERGED: 4000ps
// MERGED:   root/foo/s  0xf3
// MERGED:   root/s  0xf3
// MERGED: 5000ps
// MERGED:   root/foo/s  0xd9
// MERGED:   root/s  0xd9

// MERGEDRED: 0ps
// MERGEDRED:   root/1  0x01
// MERGEDRED:   root/s  0x03
// MERGEDRED: 1000ps
// MERGEDRED:   root/s  0x09
// MERGEDRED: 2000ps
// MERGEDRED:   root/s  0x1b
// MERGEDRED: 3000ps
// MERGEDRED:   root/s  0x51
// MERGEDRED: 4000ps
// MERGEDRED:   root/s  0xf3
// MERGEDRED: 5000ps
// MERGEDRED:   root/s  0xd9

// NAMED: 0ps
// NAMED:   root/s  0x03
// NAMED: 1000ps
// NAMED:   root/s  0x09
// NAMED: 2000ps
// NAMED:   root/s  0x1b
// NAMED: 3000ps
// NAMED:   root/s  0x51
// NAMED: 4000ps
// NAMED:   root/s  0xf3
// NAMED: 5000ps
// NAMED:   root/s  0xd9
llhd.entity @root () -> () {
  %0 = hw.constant 1 : i8
  %s = llhd.sig "s" %0 : i8
  %1 = llhd.sig "1" %0 : i8
  llhd.inst "foo" @foo () -> (%s) : () -> (!llhd.sig<i8>)
}

llhd.proc @foo () -> (%s : !llhd.sig<i8>) {
  cf.br ^entry
^entry:
  %1 = llhd.prb %s : !llhd.sig<i8>
  %2 = comb.add %1, %1 : i8
  %t0 = llhd.constant_time #llhd.time<0ns, 0d, 1e>
  llhd.drv %s, %2 after %t0 : !llhd.sig<i8>
  %3 = comb.add %2, %1 : i8
  %t1 = llhd.constant_time #llhd.time<0ns, 0d, 2e>
  llhd.drv %s, %3 after %t1 : !llhd.sig<i8>
  %t2= llhd.constant_time #llhd.time<1ns, 0d, 0e>
  llhd.wait for %t2, ^entry
}
