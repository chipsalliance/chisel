// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129

// This test checks that only the reduction patterns of dialects that occur in
// the input file are registered

// RUN: circt-reduce %s --test /usr/bin/env --test-arg cat --list | FileCheck %s

// CHECK:      firrtl-lower-chirrtl
// CHECK-NEXT: firrtl-infer-widths
// CHECK-NEXT: firrtl-infer-resets
// CHECK-NEXT: firrtl-module-externalizer
// CHECK-NEXT: instance-stubber
// CHECK-NEXT: memory-stubber
// CHECK-NEXT: eager-inliner
// CHECK-NEXT: firrtl-lower-types
// CHECK-NEXT: firrtl-expand-whens
// CHECK-NEXT: firrtl-inliner
// CHECK-NEXT: firrtl-imconstprop
// CHECK-NEXT: firrtl-remove-unused-ports
// CHECK-NEXT: node-symbol-remover
// CHECK-NEXT: connect-forwarder
// CHECK-NEXT: connect-invalidator
// CHECK-NEXT: firrtl-constantifier
// CHECK-NEXT: firrtl-operand0-forwarder
// CHECK-NEXT: firrtl-operand1-forwarder
// CHECK-NEXT: firrtl-operand2-forwarder
// CHECK-NEXT: detach-subaccesses
// CHECK-NEXT: hw-module-externalizer
// CHECK-NEXT: annotation-remover
// CHECK-NEXT: hw-constantifier
// CHECK-NEXT: root-port-pruner
// CHECK-NEXT: hw-operand0-forwarder
// CHECK-NEXT: extmodule-instance-remover
// CHECK-NEXT: cse
// CHECK-NEXT: hw-operand1-forwarder
// CHECK-NEXT: connect-source-operand-0-forwarder
// CHECK-NEXT: canonicalize
// CHECK-NEXT: hw-operand2-forwarder
// CHECK-NEXT: connect-source-operand-1-forwarder
// CHECK-NEXT: operation-pruner
// CHECK-NEXT: connect-source-operand-2-forwarder
// CHECK-NEXT: module-internal-name-sanitizer
// CHECK-NEXT: module-name-sanitizer
// CHECK-EMPTY:
firrtl.circuit "Foo" {
  firrtl.module @Foo() {}
}
