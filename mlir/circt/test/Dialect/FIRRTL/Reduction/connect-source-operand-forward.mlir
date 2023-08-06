// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129
// RUN: circt-reduce %s --test /usr/bin/env --test-arg grep --test-arg -q --test-arg "firrtl.module @Foo" --keep-best=0 --include connect-source-operand-0-forwarder | FileCheck %s
firrtl.circuit "Foo" {
  // CHECK-LABEL: firrtl.module @Foo
  firrtl.module @Foo(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %val: !firrtl.uint<2>) {
    %a = firrtl.wire : !firrtl.uint<1>
    %b = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<1>
    %c = firrtl.regreset %clock, %reset, %reset : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    %0 = firrtl.bits %val 0 to 0 : (!firrtl.uint<2>) -> !firrtl.uint<1>
    firrtl.connect %a, %0 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %b, %0 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %c, %0 : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK-NEXT:   %a = firrtl.wire  : !firrtl.uint<2>
    // CHECK-NEXT:   %b = firrtl.reg %clock  : !firrtl.clock, !firrtl.uint<2>
    // CHECK-NEXT:   %c = firrtl.reg %clock  : !firrtl.clock, !firrtl.uint<2>
    // CHECK-NEXT:   firrtl.connect %a, %val : !firrtl.uint<2>, !firrtl.uint<2>
    // CHECK-NEXT:   firrtl.connect %b, %val : !firrtl.uint<2>, !firrtl.uint<2>
    // CHECK-NEXT:   firrtl.connect %c, %val : !firrtl.uint<2>, !firrtl.uint<2>
    // CHECK-NEXT: }
  }
}
