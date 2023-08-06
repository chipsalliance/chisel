// RUN: circt-opt %s --split-input-file --verify-diagnostics

ibis.class @C {
  ibis.method @typeMismatch1() -> ui32 {
    // expected-error @+1 {{must return a value}}
    ibis.return
  }
}

// -----

ibis.class @C {
  ibis.method @typeMismatch2() {
    %c = hw.constant 1 : i8
    // expected-error @+1 {{cannot return a value from a function with no result type}}
    ibis.return %c : i8
  }
}

// -----
ibis.class @C {
  ibis.method @typeMismatch3() -> ui32 {
    %c = hw.constant 1 : i8
    // expected-error @+1 {{return type ('i8') must match function return type ('ui32')}}
    ibis.return %c : i8
  }
}
