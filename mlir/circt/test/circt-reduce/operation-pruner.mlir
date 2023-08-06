// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129
// RUN: circt-reduce %s --test /usr/bin/env --test-arg grep --test-arg -q --test-arg "hw.module @Foo" --keep-best=0 --include operation-pruner | FileCheck %s

// CHECK-LABEL: hw.module @Foo
hw.module @Foo(%arg0: i32) -> (out: i32) {
  hw.output %arg0 : i32
}

// CHECK-NOT: hw.module @Bar
hw.module @Bar(%arg0: i32) -> (out: i32) {
  hw.output %arg0 : i32
}
