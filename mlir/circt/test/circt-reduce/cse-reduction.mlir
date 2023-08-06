// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129
// RUN: circt-reduce %s --test /usr/bin/env --test-arg grep --test-arg -q --test-arg "hw.module @Foo" --keep-best=0 --include cse | FileCheck %s

// CHECK-LABEL: hw.module @Foo
hw.module @Foo() {
  // CHECK-NOT: hw.constant
  %0 = hw.constant 0 : i32
}
