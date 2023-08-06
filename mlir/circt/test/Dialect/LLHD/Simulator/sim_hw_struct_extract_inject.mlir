// REQUIRES: llhd-sim
// RUN: llhd-sim %s -shared-libs=%shlibdir/libcirct-llhd-signals-runtime-wrappers%shlibext | FileCheck %s

// This test checks correct simulation of the following operations and ensures
// that the endianess semantics as described in the rationale are followed.
// * hw.struct_create
// * hw.struct_extract
// * hw.struct_inject

// CHECK: 0ps 0d 0e  root/ext  0x00
// CHECK-NEXT: 0ps 0d 0e  root/struct  0x00000000
// CHECK-NEXT: 1000ps 0d 0e  root/ext  0x03
// CHECK-NEXT: 1000ps 0d 0e  root/struct  0x01050304
llhd.entity @root () -> () {
    %zero = hw.constant 0 : i8
    %init = hw.constant 0 : i32
    %1 = hw.constant 1 : i8
    %2 = hw.constant 2 : i8
    %3 = hw.constant 3 : i8
    %4 = hw.constant 4 : i8
    %5 = hw.constant 5 : i8

    %struct = hw.struct_create (%1, %2, %3, %4) : !hw.struct<a: i8, b: i8, c: i8, d: i8>
    %ext = hw.struct_extract %struct["c"] : !hw.struct<a: i8, b: i8, c: i8, d: i8>
    %inj = hw.struct_inject %struct["b"], %5 : !hw.struct<a: i8, b: i8, c: i8, d: i8>

    %structsig = llhd.sig "struct" %init : i32
    %extsig = llhd.sig "ext" %zero : i8

    %0 = hw.bitcast %inj : (!hw.struct<a: i8, b: i8, c: i8, d: i8>) -> i32

    %time = llhd.constant_time #llhd.time<1ns, 0d, 0e>

    llhd.drv %structsig, %0 after %time : !llhd.sig<i32>
    llhd.drv %extsig, %ext after %time : !llhd.sig<i8>
}
