// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129
// RUN: circt-reduce %s --test /usr/bin/env --test-arg grep --test-arg -q --test-arg "DummyArc(%arg0)" --keep-best=0 --include arc-state-elimination | FileCheck %s

// CHECK-LABEL: hw.module @Foo
hw.module @Foo(%clk: i1, %en: i1, %rst: i1, %arg0: i32) -> (out: i32) {
  // CHECK-NEXT: [[V0:%.+]] = arc.call @DummyArc(%arg0) : (i32) -> i32
  %0 = arc.state @DummyArc(%arg0) clock %clk enable %en reset %rst lat 1 {name="reg1"} : (i32) -> (i32)
  // CHECK-NEXT: hw.output [[V0]]
  hw.output %0 : i32
}
arc.define @DummyArc(%arg0: i32) -> i32 {
  arc.output %arg0 : i32
}
