// RUN: circt-opt --convert-hw-to-systemc --verify-diagnostics --split-input-file %s

// expected-error @+2 {{module parameters not supported yet}}
// expected-error @+1 {{failed to legalize operation 'hw.module'}}
hw.module @someModule<p1: i42 = 17, p2: i1>() -> () {}

// -----

// expected-error @+2 {{inout arguments not supported yet}}
// expected-error @+1 {{failed to legalize operation 'hw.module'}}
hw.module @someModule(%in0: !hw.inout<i32>) -> () {}

// -----

hw.module @graphRegionToSSACFG(%in0: i32) -> () {
    // expected-error @+1 {{operand #1 does not dominate this use}}
    %0 = comb.add %in0, %1 : i32
    // expected-note @+1 {{operand defined here (op in the same block)}}
    %1 = comb.add %in0, %0 : i32
}
