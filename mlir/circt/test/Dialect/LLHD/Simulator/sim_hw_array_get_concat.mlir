// REQUIRES: llhd-sim
// RUN: llhd-sim %s -shared-libs=%shlibdir/libcirct-llhd-signals-runtime-wrappers%shlibext | FileCheck %s

// This test checks correct simulation of the following operations and ensures
// that the endianess semantics as described in the rationale are followed.
// * hw.array_create
// * hw.array_slice
// * hw.array_get
// * hw.array_concat

// CHECK: 0ps 0d 0e  root/concat[0]  0x00
// CHECK-NEXT: 0ps 0d 0e  root/concat[1]  0x00
// CHECK-NEXT: 0ps 0d 0e  root/concat[2]  0x00
// CHECK-NEXT: 0ps 0d 0e  root/concat[3]  0x00
// CHECK-NEXT: 0ps 0d 0e  root/get  0x00
// CHECK-NEXT: 0ps 0d 0e  root/slice[0]  0x01
// CHECK-NEXT: 0ps 0d 0e  root/slice[1]  0x00
// CHECK-NEXT: 1000ps 0d 0e  root/concat[0]  0x03
// CHECK-NEXT: 1000ps 0d 0e  root/concat[1]  0x02
// CHECK-NEXT: 1000ps 0d 0e  root/concat[2]  0x01
// CHECK-NEXT: 1000ps 0d 0e  root/get  0x02
// CHECK-NEXT: 1000ps 0d 0e  root/slice[0]  0x03
// CHECK-NEXT: 1000ps 0d 0e  root/slice[1]  0x02
llhd.entity @root () -> () {
    %0 = hw.constant 0 : i8
    %1 = hw.constant 1 : i8
    %2 = hw.constant 2 : i8
    %3 = hw.constant 3 : i8
    %init = hw.constant 0 : i32
    %index = hw.constant 1 : i2
    %indexz = hw.constant 0 : i2

    %arrayinit = hw.array_create %0, %0, %0, %0 : i8
    %array1 = hw.array_create %0, %1 : i8
    %array2 = hw.array_create %2, %3 : i8

    %array = hw.array_concat %array1, %array2 : !hw.array<2xi8>, !hw.array<2xi8>
    %get = hw.array_get %array[%index] : !hw.array<4xi8>, i2
    %slice = hw.array_slice %array[%indexz] : (!hw.array<4xi8>) -> !hw.array<2xi8>

    %concatsig = llhd.sig "concat" %arrayinit : !hw.array<4xi8> 
    %getsig = llhd.sig "get" %0 : i8
    %slicesig = llhd.sig "slice" %array1 : !hw.array<2xi8>

    %time = llhd.constant_time #llhd.time<1ns, 0d, 0e>

    llhd.drv %concatsig, %array after %time : !llhd.sig<!hw.array<4xi8>>
    llhd.drv %getsig, %get after %time : !llhd.sig<i8>
    llhd.drv %slicesig, %slice after %time : !llhd.sig<!hw.array<2xi8>>
}
