// UNSUPPORTED: system-windows
// RUN: circt-reduce %s --include=module-internal-name-sanitizer --include=module-name-sanitizer --test /usr/bin/env --test-arg true --keep-best=0 | FileCheck %s

// CHECK-LABEL: firrtl.circuit "Foo"
firrtl.circuit "A" {
  // CHECK-NEXT: firrtl.module private @Bar
  // CHECK-SAME:   in %clk: !firrtl.clock
  // CHECK-SAME:   in %clk_0: !firrtl.clock
  // CHECK-SAME:   in %rst: !firrtl.reset
  // CHECK-SAME:   in %rst_0: !firrtl.reset
  // CHECK-SAME:   out %ref: !firrtl.probe<uint<1>>
  // CHECK-SAME:   out %ref_0: !firrtl.rwprobe<uint<1>>
  // CHECK-SAME:   in %a: !firrtl.uint<1>
  // CHECK-SAME:   out %b: !firrtl.uint<1>
  firrtl.module private @B(
    in %clock: !firrtl.clock,
    in %clock2: !firrtl.clock,
    in %reset: !firrtl.reset,
    in %reset2: !firrtl.reset,
    out %someProbe: !firrtl.probe<uint<1>>,
    out %someOtherProbe: !firrtl.rwprobe<uint<1>>,
    in %x: !firrtl.uint<1>,
    out %y: !firrtl.uint<1>
  ) {
    // CHECK-NEXT: %reg = firrtl.reg
    // CHECK:      firrtl.regreset
    // CHECK-SAME:   {name = "reg"}
    %derp = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<1>
    %c1_ui1 = firrtl.constant 1 : !firrtl.const.uint<1>
    %herp = firrtl.regreset %clock, %reset, %c1_ui1 : !firrtl.clock, !firrtl.reset, !firrtl.const.uint<1>, !firrtl.uint<1>
  }
  // CHECK:      firrtl.module @Foo
  // CHECK-SAME:   in %clk: !firrtl.clock
  // CHECK-SAME:   in %a: !firrtl.uint<1>
  // CHECK-SAME:   in %rst: !firrtl.asyncreset
  // CHECK-SAME:   out %b: !firrtl.uint<1>
  // CHECK-SAME:   out %ref: !firrtl.probe<uint<1>>
  // CHECK-SAME:   out %ref_0: !firrtl.rwprobe<uint<1>>
  firrtl.module @A(
    in %clock: !firrtl.clock,
    in %reset: !firrtl.uint<1>,
    in %reset2: !firrtl.asyncreset,
    out %out: !firrtl.uint<1>,
    out %aProbe: !firrtl.probe<uint<1>>,
    out %bProbe: !firrtl.rwprobe<uint<1>>
  ) attributes {convention = #firrtl<convention scalarized>} {
    // CHECK-NEXT: %wire = firrtl.wire
    // CHECK-NEXT: firrtl.wire {name = "wire"}
    %foo = firrtl.wire : !firrtl.uint<1>
    %bar = firrtl.wire : !firrtl.uint<1>
    // CHECK-NEXT: %node = firrtl.node
    // CHECK-NEXT: firrtl.node {{.*}} {name = "node"}
    %baz = firrtl.node %bar : !firrtl.uint<1>
    %qux = firrtl.node %baz : !firrtl.uint<1>
    %b_clock, %b_clock2, %b_reset, %b_reset2,  %b_someProbe, %b_someOtherProbe,
      %b_x, %b_y = firrtl.instance b @B(
        in clock: !firrtl.clock,
        in clock2: !firrtl.clock,
        in reset: !firrtl.reset,
        in reset2: !firrtl.reset,
        out someProbe: !firrtl.probe<uint<1>>,
        out someOtherProbe: !firrtl.rwprobe<uint<1>>,
        in x: !firrtl.uint<1>,
        out y: !firrtl.uint<1>
      )
  }
}
