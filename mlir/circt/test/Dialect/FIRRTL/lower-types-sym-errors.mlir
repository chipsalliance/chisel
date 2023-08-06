// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-types))' %s --verify-diagnostics

// Check diagnostic when attempting to lower something with symbols on it.
firrtl.circuit "InnerSym" {
  // expected-error @below {{unable to lower due to symbol "x" with target not preserved by lowering}}
  firrtl.module @InnerSym(in %x: !firrtl.bundle<a: uint<5>, b: uint<3>> sym @x) { }
}
