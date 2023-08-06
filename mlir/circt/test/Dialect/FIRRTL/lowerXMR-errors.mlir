// RUN: circt-opt %s  --firrtl-lower-xmr -split-input-file -verify-diagnostics

// Test for same module lowering
// CHECK-LABEL: firrtl.circuit "xmr"
firrtl.circuit "xmr" {
  // expected-error @+1 {{reference dataflow cannot be traced back to the remote read op for module port 'a'}}
  firrtl.module @xmr(in %a: !firrtl.probe<uint<2>>) {
    %x = firrtl.ref.resolve %a : !firrtl.probe<uint<2>>
  }
}

// -----

firrtl.circuit "Top" {
  firrtl.module @XmrSrcMod(out %_a: !firrtl.probe<uint<1>>) {
    %zero = firrtl.constant 0 : !firrtl.uint<1>
    %1 = firrtl.ref.send %zero : !firrtl.uint<1>
    firrtl.ref.define %_a, %1 : !firrtl.probe<uint<1>>
  }
  firrtl.module @Top() {
    %xmr_a = firrtl.instance xmr sym @xmr @XmrSrcMod(out _a: !firrtl.probe<uint<1>>)
    %c_a = firrtl.instance child @Child1(in _a: !firrtl.probe<uint<1>>)
    %c_b = firrtl.instance child @Child2(in _a: !firrtl.probe<uint<1>>)
    firrtl.ref.define %c_a, %xmr_a : !firrtl.probe<uint<1>>
    firrtl.ref.define %c_b, %xmr_a : !firrtl.probe<uint<1>>
  }
  firrtl.module @Child1(in  %_a: !firrtl.probe<uint<1>>) {
    %0 = firrtl.ref.resolve %_a : !firrtl.probe<uint<1>>
    %c_b = firrtl.instance child @Child2(in _a: !firrtl.probe<uint<1>>)
    firrtl.ref.define %c_b, %_a : !firrtl.probe<uint<1>>
  }
  // expected-error @+1 {{op multiply instantiated module with input RefType port '_a'}}
  firrtl.module @Child2(in  %_a: !firrtl.probe<uint<1>>) {
    %0 = firrtl.ref.resolve %_a : !firrtl.probe<uint<1>>
  }
}

// -----

firrtl.circuit "Top" {
  firrtl.module @XmrSrcMod(out %_a: !firrtl.probe<uint<1>>) {
    %zero = firrtl.constant 0 : !firrtl.uint<1>
    %1 = firrtl.ref.send %zero : !firrtl.uint<1>
    firrtl.ref.define %_a, %1 : !firrtl.probe<uint<1>>
  }
  firrtl.module @Top() {
    %xmr_a = firrtl.instance xmr sym @xmr @XmrSrcMod(out _a: !firrtl.probe<uint<1>>)
    %c_a = firrtl.instance child @Child1(in _a: !firrtl.probe<uint<1>>)
    %c_b = firrtl.instance child @Child2(in _a: !firrtl.probe<uint<1>>)
    firrtl.ref.define %c_a, %xmr_a : !firrtl.probe<uint<1>>
  }
  firrtl.module @Child1(in  %_a: !firrtl.probe<uint<1>>) {
    %0 = firrtl.ref.resolve %_a : !firrtl.probe<uint<1>>
  }
  // expected-error @+1 {{reference dataflow cannot be traced back to the remote read op for module port '_a'}}
  firrtl.module @Child2(in  %_a: !firrtl.probe<uint<1>>) {
    %0 = firrtl.ref.resolve %_a : !firrtl.probe<uint<1>>
  }
}

// -----
// Check handling of unexpected ref.sub, from input port.

firrtl.circuit "RefSubNotFromOp" {
  // expected-error @below {{reference dataflow cannot be traced back to the remote read op for module port 'ref'}}
  // expected-note @below {{indexing through this reference}}
  firrtl.module private @Child(in %ref : !firrtl.probe<bundle<a: uint<1>, b: uint<2>>>) {
    // expected-error @below {{indexing through probe of unknown origin (input probe?)}}
    %sub = firrtl.ref.sub %ref[1] : !firrtl.probe<bundle<a: uint<1>, b: uint<2>>>
    %res = firrtl.ref.resolve %sub : !firrtl.probe<uint<2>>
  }
  firrtl.module @RefSubNotFromOp(in %in : !firrtl.bundle<a: uint<1>, b: uint<2>>) {
    %ref = firrtl.ref.send %in : !firrtl.bundle<a: uint<1>, b: uint<2>>
    %child_ref = firrtl.instance child @Child(in ref : !firrtl.probe<bundle<a: uint<1>, b: uint<2>>>)
    firrtl.ref.define %child_ref, %ref : !firrtl.probe<bundle<a: uint<1>, b: uint<2>>>
  }
}
