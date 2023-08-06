// RUN: circt-opt -hw-stub-external-modules %s | FileCheck %s

// CHECK-LABEL: hw.module @local(%arg0: !hw.inout<i1>, %arg1: i1) -> (r: i3, s: i3) {
// CHECK-NEXT: %foo.r = hw.instance "foo" @remote(arg0: %arg0: !hw.inout<i1>, arg1: %arg1: i1) -> (r: i3)
// CHECK-NEXT: %foo.r_0 = hw.instance "foo" @remoteWithParams<a: i4 = 3>(arg0: %arg0: !hw.inout<i1>, arg1: %arg1: i1) -> (r: i3)
// CHECK-NEXT: hw.output %foo.r, %foo.r_0 : i3, i3
// CHECK-NEXT:  }
// CHECK-LABEL: hw.module @remote(%arg0: !hw.inout<i1>, %arg1: i1) -> (r: i3) {
// CHECK-NEXT: %x_i3 = sv.constantX : i3
// CHECK-NEXT:   hw.output %x_i3 : i3
// CHECK-NEXT: }
// CHECK-LABEL: hw.module @remoteWithParams<a: i4>(%arg0: !hw.inout<i1>, %arg1: i1) -> (r: i3) {
// CHECK-NEXT: %x_i3 = sv.constantX : i3
// CHECK-NEXT:   hw.output %x_i3 : i3
// CHECK-NEXT: }

hw.module.extern @remote(%arg0: !hw.inout<i1>, %arg1: i1) -> (r : i3)

hw.module.extern @remoteWithParams<a: i4>(%arg0: !hw.inout<i1>, %arg1: i1) -> (r : i3)


hw.module @local(%arg0: !hw.inout<i1>, %arg1: i1) -> (r: i3, s: i3) {
    %tr = hw.instance "foo" @remote(arg0: %arg0: !hw.inout<i1>, arg1: %arg1: i1) -> (r: i3)

    %tr2 = hw.instance "foo" @remoteWithParams<a: i4 = 3>(arg0: %arg0: !hw.inout<i1>, arg1: %arg1: i1) -> (r: i3)
    hw.output %tr, %tr2 : i3, i3
}
