// RUN: circt-opt --pass-pipeline="builtin.module(firrtl.circuit(firrtl-lower-open-aggs))" %s --split-input-file --verify-diagnostics

firrtl.circuit "Symbol" {
  // expected-error @below {{inner symbol "bad" mapped to non-HW type}}
  firrtl.module @Symbol(out %r : !firrtl.openbundle<p: probe<uint<1>>> sym @bad) {
    %zero = firrtl.constant 0 : !firrtl.uint<1>
    %ref = firrtl.ref.send %zero : !firrtl.uint<1>
    %r_p = firrtl.opensubfield %r[p] : !firrtl.openbundle<p: probe<uint<1>>>
    firrtl.ref.define %r_p, %ref : !firrtl.probe<uint<1>>
  }
}

// -----

firrtl.circuit "SymbolOnField" {
  // expected-error @below {{inner symbol "bad" mapped to non-HW type}}
  firrtl.extmodule @SymbolOnField(out r : !firrtl.openbundle<p: probe<uint<1>>, x: uint<1>> sym [<@bad,1,public>])
}

// -----

firrtl.circuit "Annotation" {
  // expected-error @below {{annotations found on aggregate with no HW}}
  firrtl.module @Annotation(out %r : !firrtl.openbundle<p: probe<uint<1>>>) attributes {portAnnotations = [[{class = "circt.test"}]]} {
    %zero = firrtl.constant 0 : !firrtl.uint<1>
    %ref = firrtl.ref.send %zero : !firrtl.uint<1>
    %r_p = firrtl.opensubfield %r[p] : !firrtl.openbundle<p: probe<uint<1>>>
    firrtl.ref.define %r_p, %ref : !firrtl.probe<uint<1>>
  }
}

// -----
// Open aggregates are expected to be removed before annotations,
// but check this is detected and an appropriate diagnostic is presented.

firrtl.circuit "MixedAnnotation" {
  // expected-error @below {{annotations on open aggregates not handled yet}}
  firrtl.module @MixedAnnotation(out %r : !firrtl.openbundle<a: uint<1>, p: probe<uint<1>>>) attributes {portAnnotations = [[{class = "circt.test"}]]} {
    %zero = firrtl.constant 0 : !firrtl.uint<1>
    %ref = firrtl.ref.send %zero : !firrtl.uint<1>
    %r_p = firrtl.opensubfield %r[p] : !firrtl.openbundle<a: uint<1>, p: probe<uint<1>>>
    firrtl.ref.define %r_p, %ref : !firrtl.probe<uint<1>>
    %r_a = firrtl.opensubfield %r[a] : !firrtl.openbundle<a: uint<1>, p: probe<uint<1>>>
    firrtl.strictconnect %r_a, %zero : !firrtl.uint<1>
  }
}

// -----
// Reject unhandled ops w/open types in them.

firrtl.circuit "UnhandledOp" {
  firrtl.module @UnhandledOp(out %r : !firrtl.openbundle<p: probe<uint<1>>>) {
    %zero = firrtl.constant 0 : !firrtl.uint<1>
    %ref = firrtl.ref.send %zero : !firrtl.uint<1>
    %r_p = firrtl.opensubfield %r[p] : !firrtl.openbundle<p: probe<uint<1>>>
    firrtl.ref.define %r_p, %ref : !firrtl.probe<uint<1>>

    // expected-error @below {{unhandled use or producer of types containing references}}
    %x = firrtl.wire : !firrtl.openbundle<p : probe<uint<1>>>
  }
}
