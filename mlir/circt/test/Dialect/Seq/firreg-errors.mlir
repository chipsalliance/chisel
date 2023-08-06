// RUN: circt-opt %s -verify-diagnostics --split-input-file

hw.module @NeedsBothResetAndResetValue(%input: i1, %clk: i1) {
  // expected-error@+1 {{'seq.firreg' op must specify reset and reset value}}
  "seq.firreg"(%input, %clk) { name = "reg", isAsync } : (i1, i1) -> i1
}
