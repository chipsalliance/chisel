// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129
// RUN: circt-reduce %s --test /usr/bin/env --test-arg grep --test-arg -q --test-arg "hw.instance" --keep-best=0 --include hw-module-externalizer | FileCheck %s

// CHECK-LABEL: hw.module @Foo
hw.module @Foo(%arg0: i32) -> (out: i32) {
  // CHECK-NEXT: hw.instance
  %inst.out = hw.instance "inst" @Bar (arg0: %arg0: i32) -> (out: i32)
  // CHECK-NEXT: hw.output
  hw.output %inst.out : i32
  // CHECK-NEXT: }
}

// CHECK-NEXT: hw.module.extern @Bar
// CHECK-NOT: hw.module @Bar
hw.module @Bar(%arg0: i32) -> (out: i32) {
  hw.output %arg0 : i32
}
