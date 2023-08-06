// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129

// This test checks that only the reduction patterns of dialects that occur in
// the input file are registered

// RUN: circt-reduce %s --test /usr/bin/env --test-arg cat --list | FileCheck %s

// CHECK: arc-strip-sv
// CHECK-NEXT: cse
// CHECK-NEXT: arc-dedup
// CHECK-NEXT: canonicalize
// CHECK-NEXT: arc-state-elimination
// CHECK-NEXT: operation-pruner
// CHECK-NEXT: arc-canonicalizer
// CHECK-EMPTY:
arc.define @DummyArc(%arg0: i32) -> i32 {
  arc.output %arg0 : i32
}

