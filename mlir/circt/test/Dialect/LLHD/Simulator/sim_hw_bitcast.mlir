// REQUIRES: llhd-sim
// RUN: llhd-sim %s -shared-libs=%shlibdir/libcirct-llhd-signals-runtime-wrappers%shlibext | FileCheck %s

// This test checks correct simulation of the following operations and ensures
// that the endianess semantics as described in the rationale are followed.
// * hw.array_create
// * hw.struct_create
// * hw.bitcast

// CHECK: 0ps 0d 0e  root/arr  0xffff0000
// CHECK-NEXT: 0ps 0d 0e  root/int[0]  0xffff
// CHECK-NEXT: 0ps 0d 0e  root/int[1]  0x0000
// CHECK-NEXT: 0ps 0d 0e  root/struct  0xffff0000
// CHECK-NEXT: 1000ps 0d 0e  root/arr  0x0000ffff
// CHECK-NEXT: 1000ps 0d 0e  root/int[0]  0x0000
// CHECK-NEXT: 1000ps 0d 0e  root/int[1]  0xffff
// CHECK-NEXT: 1000ps 0d 0e  root/struct  0x0000ff00
llhd.entity @root () -> () {
    %allset = hw.constant 0xffff : i16
    %allset2 = hw.constant 0xff : i8
    %zero = hw.constant 0 : i16
    %zero2 = hw.constant 0 : i8
    %init = hw.constant 0xffff0000 : i32

    %arr = hw.array_create %zero, %allset : i16
    %struct = hw.struct_create (%zero, %allset2, %zero2) : !hw.struct<foo: i16, bar: i8, baz: i8>

    %arrsig = llhd.sig "arr" %init : i32
    %structsig = llhd.sig "struct" %init : i32
    %intsig = llhd.sig "int" %arr : !hw.array<2xi16>

    %0 = hw.bitcast %init : (i32) -> !hw.array<2xi16>
    %1 = hw.bitcast %arr : (!hw.array<2xi16>) -> i32
    %2 = hw.bitcast %struct : (!hw.struct<foo: i16, bar: i8, baz: i8>) -> i32

    %time = llhd.constant_time #llhd.time<1ns, 0d, 0e>

    llhd.drv %intsig, %0 after %time : !llhd.sig<!hw.array<2xi16>>
    llhd.drv %arrsig, %1 after %time : !llhd.sig<i32>
    llhd.drv %structsig, %2 after %time : !llhd.sig<i32>
}
