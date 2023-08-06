// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129
// RUN: circt-reduce %s --test /usr/bin/env --test-arg grep --test-arg -q --test-arg "hw.module @Foo" --keep-best=0 --include canonicalize | FileCheck %s

// CHECK-LABEL: hw.module @Foo
hw.module @Foo(%arg0: i32) -> (out: i32) {
  %0 = comb.and %arg0, %arg0 : i32
  // CHECK: hw.output %arg0 :
  hw.output %0 : i32
}
