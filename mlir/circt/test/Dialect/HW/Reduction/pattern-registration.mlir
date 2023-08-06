// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129

// This test checks that only the reduction patterns of dialects that occur in
// the input file are registered

// RUN: circt-reduce %s --test /usr/bin/env --test-arg cat --list | FileCheck %s

// CHECK:      hw-module-externalizer
// CHECK-NEXT: hw-constantifier
// CHECK-NEXT: hw-operand0-forwarder
// CHECK-NEXT: cse
// CHECK-NEXT: hw-operand1-forwarder
// CHECK-NEXT: canonicalize
// CHECK-NEXT: hw-operand2-forwarder
// CHECK-NEXT: operation-pruner
// CHECK-EMPTY:
hw.module @Foo(%in: i1) -> (out: i1) {
  hw.output %in : i1
}
