// RUN: circt-opt -convert-hw-to-llhd -split-input-file -verify-diagnostics %s

module {
  // Since HW-to-LLHD needs to construct a zero value for temporary signals,
  // we don't support non-IntegerType arguments to instances.
  hw.module @sub(%in: f16) -> (out: f16) {
    hw.output %in: f16
  }
  hw.module @test(%in: f16) -> (out: f16) {
    // expected-error @+1 {{failed to legalize operation 'hw.instance'}}
    %0 = hw.instance "sub1" @sub (in: %in: f16) -> (out: f16)
    hw.output %0: f16
  }
}
