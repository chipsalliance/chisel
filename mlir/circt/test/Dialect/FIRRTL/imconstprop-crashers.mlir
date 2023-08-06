// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-imconstprop))' -verify-diagnostics --split-input-file %s


// https://github.com/llvm/circt/issues/1187
// This shouldn't crash.
firrtl.circuit "Issue1187"  {
  firrtl.module @Issue1187(in %divisor: !firrtl.uint<1>, out %result: !firrtl.uint<0>) {
    %dividend = firrtl.wire  : !firrtl.uint<0>
    %invalid_ui0 = firrtl.invalidvalue : !firrtl.uint<0>
    firrtl.strictconnect %dividend, %invalid_ui0 : !firrtl.uint<0>
    %0 = firrtl.div %dividend, %divisor : (!firrtl.uint<0>, !firrtl.uint<1>) -> !firrtl.uint<0>
    firrtl.strictconnect %result, %0 : !firrtl.uint<0>
  }
}

// -----

// https://github.com/llvm/circt/issues/4456
// This shouldn't crash.
// The fold is what should be tested but need IMCP to drive it.
firrtl.circuit "Issue4456"  {
  firrtl.module @Issue4456(in %i: !firrtl.sint<0>, out %o: !firrtl.uint<4>) {
    %c0_si4 = firrtl.constant 0 : !firrtl.sint<4>
    %0 = firrtl.cat %i, %c0_si4 : (!firrtl.sint<0>, !firrtl.sint<4>) -> !firrtl.uint<4>
    firrtl.strictconnect %o, %0 : !firrtl.uint<4>
  }
}
