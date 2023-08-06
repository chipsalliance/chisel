// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-register-optimizer)))' %s | FileCheck %s

firrtl.circuit "invalidReg"   {
  // CHECK-LABEL: @invalidReg
  firrtl.module @invalidReg(in %clock: !firrtl.clock, out %a: !firrtl.uint<1>) {
    %foobar = firrtl.reg %clock  : !firrtl.clock, !firrtl.uint<1>
    firrtl.strictconnect %foobar, %foobar : !firrtl.uint<1>
    //CHECK-NOT: firrtl.connect %foobar, %foobar
    //CHECK: %[[inv:.*]] = firrtl.invalidvalue
    //CHECK: firrtl.strictconnect %a, %[[inv]]
    firrtl.strictconnect %a, %foobar : !firrtl.uint<1>
  }

  // CHECK-LABEL: @constantRegWrite
  firrtl.module @constantRegWrite(in %clock: !firrtl.clock, out %a: !firrtl.uint<1>) {
    %c = firrtl.constant 0 : !firrtl.uint<1>
    %foobar = firrtl.reg %clock  : !firrtl.clock, !firrtl.uint<1>
    firrtl.strictconnect %foobar, %c : !firrtl.uint<1>
    //CHECK-NOT: firrtl.connect %foobar, %c
    //CHECK: %[[const:.*]] = firrtl.constant
    //CHECK: firrtl.strictconnect %a, %[[const]]
    firrtl.strictconnect %a, %foobar : !firrtl.uint<1>
  }

  // CHECK-LABEL: @constantRegWriteDom
  firrtl.module @constantRegWriteDom(in %clock: !firrtl.clock, out %a: !firrtl.uint<1>) {
    %foobar = firrtl.reg %clock  : !firrtl.clock, !firrtl.uint<1>
    //CHECK-NOT: firrtl.connect %foobar, %c
    //CHECK: %[[const:.*]] = firrtl.constant
    //CHECK: firrtl.strictconnect %a, %[[const]]
    firrtl.strictconnect %a, %foobar : !firrtl.uint<1>
    %c = firrtl.constant 0 : !firrtl.uint<1>
    firrtl.strictconnect %foobar, %c : !firrtl.uint<1>
  }

  // CHECK-LABEL: @constantRegResetWrite
  firrtl.module @constantRegResetWrite(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, out %a: !firrtl.uint<1>) {
    %c = firrtl.constant 0 : !firrtl.uint<1>
    %foobar = firrtl.regreset %clock, %reset, %c  : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.strictconnect %foobar, %c : !firrtl.uint<1>
    //CHECK-NOT: firrtl.connect %foobar, %c
    //CHECK: %[[const:.*]] = firrtl.constant
    //CHECK: firrtl.strictconnect %a, %[[const]]
    firrtl.strictconnect %a, %foobar : !firrtl.uint<1>
  }

  // CHECK-LABEL: @constantRegResetWriteSelf
  firrtl.module @constantRegResetWriteSelf(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, out %a: !firrtl.uint<1>) {
    %c = firrtl.constant 0 : !firrtl.uint<1>
    %foobar = firrtl.regreset %clock, %reset, %c  : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.strictconnect %foobar, %foobar : !firrtl.uint<1>
    //CHECK-NOT: firrtl.connect %foobar, %c
    //CHECK: %[[const:.*]] = firrtl.constant
    //CHECK: firrtl.strictconnect %a, %[[const]]
    firrtl.strictconnect %a, %foobar : !firrtl.uint<1>
  }

  // CHECK-LABEL: @movedFromIMCP
  firrtl.module @movedFromIMCP(
        in %clock: !firrtl.clock,
        in %reset: !firrtl.uint<1>,
        out %result6: !firrtl.uint<2>,
        out %result7: !firrtl.uint<4>) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %c0_ui2 = firrtl.constant 0 : !firrtl.uint<2>
    %c0_ui4 = firrtl.constant 0 : !firrtl.uint<4>
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>

    // regreset
    %regreset = firrtl.regreset %clock, %reset, %c0_ui2 : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<2>

    firrtl.strictconnect %regreset, %c0_ui2 : !firrtl.uint<2>

    // CHECK: firrtl.strictconnect %result6, %c0_ui2
    firrtl.strictconnect %result6, %regreset: !firrtl.uint<2>

    // reg
    %reg = firrtl.reg %clock  : !firrtl.clock, !firrtl.uint<4>
    firrtl.strictconnect %reg, %c0_ui4 : !firrtl.uint<4>
    // CHECK: firrtl.strictconnect %result7, %c0_ui4
    firrtl.strictconnect %result7, %reg: !firrtl.uint<4>
  }

  // CHECK-LABEL: RegResetImplicitExtOrTrunc
  firrtl.module @RegResetImplicitExtOrTrunc(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, out %out: !firrtl.uint<4>) {
    // CHECK: firrtl.regreset
    %c0_ui3 = firrtl.constant 0 : !firrtl.uint<3>
    %r = firrtl.regreset %clock, %reset, %c0_ui3 : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<3>, !firrtl.uint<2>
    %0 = firrtl.cat %r, %r : (!firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<4>
    firrtl.strictconnect %r, %r : !firrtl.uint<2>
    firrtl.strictconnect %out, %0 : !firrtl.uint<4>
  }
}
