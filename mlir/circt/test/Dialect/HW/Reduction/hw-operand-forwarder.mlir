// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129
// RUN: circt-reduce %s --test /usr/bin/env --test-arg grep --test-arg -q --test-arg "hw.module @Foo" --keep-best=0 --include hw-operand0-forwarder | FileCheck %s

// CHECK-LABEL: hw.module @Foo
hw.module @Foo(%arg0: i32, %arg1: i32) -> (out: i32) {
  // COM: operand 0 is forwarded here
  %0 = comb.and %arg0, %arg1 : i32
  // CHECK-NEXT: [[V0:%.+]] = comb.and [[V0]], %arg0 : i32
  // COM: cannot forward operand 0 here as it forms a loop
  %1 = comb.and %1, %0 : i32
  // CHECK-NEXT: hw.output [[V0]]
  hw.output %1 : i32
}
