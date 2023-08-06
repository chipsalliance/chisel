// Check behavior when attempting to "rematerialize" aggregates that
// have their producers lowered but have active users still.

// This test relies on the scalarization to expand the aggregates fully
// in the ports and checks that at various preservation levels this
// 1)does not crash, and 2)appropriate aggregates are recreated.

// Derived from this example:
// https://github.com/llvm/circt/pull/5051/files#r1178556059

// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-types))' %s | FileCheck --check-prefixes=CHECK,COMMON --implicit-check-not=mux %s
// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-types{preserve-aggregate=1d-vec}))' %s | FileCheck --check-prefixes=1DVEC,COMMON --implicit-check-not=mux %s
// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-types{preserve-aggregate=all}))' %s | FileCheck --check-prefixes=ALL,COMMON --implicit-check-not=mux %s

// COMMON-LABEL: firrtl.circuit "Bar"
firrtl.circuit "Bar" {
// COMMON-LABEL: firrtl.module @Bar
// COMMON-SAME: %a2_a_1

// No agg preservation:
// CHECK-COUNT-3: firrtl.mux
// CHECK-COUNT-3: firrtl.strictconnect

// 1d-vec preservation:
// (recreate the vector leaves, but that's it)
// 1DVEC-COUNT-2: vectorcreate
// 1DVEC-COUNT-2: mux
// 1DVEC-COUNT-2: subindex

// All-agg preservation:
// Single mux, recreate fully.
// ALL:      vectorcreate
// ALL-NEXT: bundlecreate
// ALL-NEXT: vectorcreate
// ALL-NEXT: bundlecreate
// ALL-NEXT: mux
  firrtl.module @Bar(in %a1: !firrtl.bundle<a: vector<uint<1>, 2>, b: uint<2>>, in %a2: !firrtl.bundle<a: vector<uint<1>, 2>, b: uint<2>>, in %cond: !firrtl.uint<1>, out %b: !firrtl.bundle<a: vector<uint<1>, 2>, b: uint<2>>) attributes {convention = #firrtl<convention scalarized>} {
    %0 = firrtl.mux(%cond, %a1, %a2) : (!firrtl.uint<1>, !firrtl.bundle<a: vector<uint<1>, 2>, b: uint<2>>, !firrtl.bundle<a: vector<uint<1>, 2>, b: uint<2>>) -> !firrtl.bundle<a: vector<uint<1>, 2>, b: uint<2>>
    firrtl.strictconnect %b, %0 : !firrtl.bundle<a: vector<uint<1>, 2>, b: uint<2>>
  }
}
