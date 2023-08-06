// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-imconstprop), canonicalize{top-down region-simplify}, firrtl.circuit(firrtl.module(firrtl-register-optimizer)))'  %s | FileCheck %s
// XFAIL: *

// These depend on more than constant prop.  They need to move.

  // CHECK-LABEL: firrtl.module @padZeroReg
  firrtl.module @padZeroReg(in %clock: !firrtl.clock, out %z: !firrtl.uint<16>) {
      %_r = firrtl.reg droppable_name %clock  :  !firrtl.uint<8>
      firrtl.strictconnect %_r, %_r : !firrtl.uint<8>
      %c171_ui8 = firrtl.constant 171 : !firrtl.uint<8>
      %_n = firrtl.node droppable_name %c171_ui8  : !firrtl.uint<8>
      %1 = firrtl.cat %_n, %_r : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<16>
      firrtl.strictconnect %z, %1 : !firrtl.uint<16>
    // CHECK: %[[TMP:.+]] = firrtl.constant 43776 : !firrtl.uint<16>
    // CHECK-NEXT: firrtl.strictconnect %z, %[[TMP]] : !firrtl.uint<16>
  }
