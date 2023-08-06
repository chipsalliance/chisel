// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129
// RUN: circt-reduce %s --test /usr/bin/env --test-arg grep --test-arg -q --test-arg "hw.module @Foo" --keep-best=0 --include hw-constantifier | FileCheck %s

// CHECK-LABEL: hw.module @Foo
hw.module @Foo(%arg0: i32, %arg1: i32) -> (out0: i32, out1: i32) {
  // CHECK-NEXT: [[V0:%.+]] = hw.constant 0
  // CHECK-NEXT: [[V1:%.+]] = hw.constant 0
  %inst.out0, %inst.out1 = hw.instance "inst" @Bar (arg0: %arg0: i32, arg1: %arg1: i32) -> (out0: i32, out1: i32)
  // CHECK-NEXT: hw.output [[V0]], [[V1]]
  hw.output %inst.out0, %inst.out1 : i32, i32
}

// CHECK: hw.module @Bar
hw.module @Bar(%arg0: i32, %arg1: i32) -> (out0: i32, out1: i32) {
  hw.output %arg0, %arg1 : i32, i32
}


// CHECK-LABEL: hw.module @FooFoo
hw.module @FooFoo(%arg0: i32, %arg1: !hw.array<2xi32>) -> (out0: i32, out1: !hw.array<2xi32>) {
  // CHECK-NEXT: [[V0:%.+]], [[V1:%.+]] = hw.instance
  %inst.out0, %inst.out1 = hw.instance "inst" @FooBar (arg0: %arg0: i32, arg1: %arg1: !hw.array<2xi32>) -> (out0: i32, out1: !hw.array<2xi32>)
  // CHECK-NEXT: hw.output [[V0]], [[V1]]
  hw.output %inst.out0, %inst.out1 : i32, !hw.array<2xi32>
}

// CHECK: hw.module @FooBar
hw.module @FooBar(%arg0: i32, %arg1: !hw.array<2xi32>) -> (out0: i32, out1: !hw.array<2xi32>) {
  hw.output %arg0, %arg1 : i32, !hw.array<2xi32>
}
