// RUN: circt-opt %s --arc-lower-clocks-to-funcs --verify-diagnostics

arc.model "NonConstExternalValue" {
^bb0(%arg0: !arc.storage<42>):
  %c0_i9001 = hw.constant 0 : i9001
  // expected-note @+1 {{external value defined here:}}
  %0 = comb.add %c0_i9001, %c0_i9001 : i9001
  // expected-note @+1 {{clock tree:}}
  arc.passthrough {
    // expected-error @+2 {{operation in clock tree uses external value}}
    // expected-note @+1 {{clock trees can only use external constant values}}
    %1 = comb.sub %0, %0 : i9001
  }
}
