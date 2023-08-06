// RUN: circt-opt %s -split-input-file -verify-diagnostics

hw.module @tooFewOps() {
  %c1_1 = hwarith.constant 1 : ui1
  // expected-error @+1 {{expected 2 operands but got 1}}
  %0 = hwarith.add %c1_1: (ui1) -> ui1
}

// -----

hw.module @tooManyOps() {
  %c1_1 = hwarith.constant 1 : ui1
  // expected-error @+1 {{expected 2 operands but got 3}}
  %0 = hwarith.add %c1_1, %c1_1, %c1_1: (ui1, ui1, ui1) -> ui2
}
