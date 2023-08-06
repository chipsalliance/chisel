// REQUIRES: llhd-sim
// RUN: llhd-sim %s -shared-libs=%shlibdir/libcirct-llhd-signals-runtime-wrappers%shlibext | FileCheck %s

// This test checks correct simulation of the following operations and ensures
// that the endianess semantics as described in the rationale are followed.
// * comb.concat

// CHECK: 0ps 0d 0e  root/int  0x00000000
// CHECK-NEXT: 1000ps 0d 0e  root/int  0x01020304
llhd.entity @root () -> () {
    %init = hw.constant 0 : i32
    %0 = hw.constant 1 : i8
    %1 = hw.constant 2 : i8
    %2 = hw.constant 3 : i8
    %3 = hw.constant 4 : i8

    %con = comb.concat %0, %1, %2, %3 : i8,i8,i8,i8
    %intsig = llhd.sig "int" %init : i32

    %time = llhd.constant_time #llhd.time<1ns, 0d, 0e>

    llhd.drv %intsig, %con after %time : !llhd.sig<i32>
}
