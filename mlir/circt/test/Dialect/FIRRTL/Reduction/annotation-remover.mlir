// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129
// RUN: circt-reduce %s --test /usr/bin/env --test-arg grep --test-arg -q --test-arg "%anotherWire = firrtl.wire" --keep-best=0 --include annotation-remover | FileCheck %s

firrtl.circuit "Foo" {
  // CHECK: firrtl.module @Foo
  // CHECK: %anotherWire = firrtl.wire
  // CHECK-NOT: annotations
  firrtl.module @Foo() {
    %oneWire = firrtl.wire : !firrtl.uint<1>
    %anotherWire = firrtl.wire {annotations = [{a}]} : !firrtl.uint<1>
  }
}
