// RUN: circt-opt %s -split-input-file -verify-diagnostics

// expected-error @+1 {{must contain exactly one 'library' op and one 'graph' op}}
ssp.instance @containers_empty of "Problem" {}

// -----

// expected-error @+1 {{must contain the 'library' op followed by the 'graph' op}}
ssp.instance @containers_wrong_order of "Problem" {
  graph {}
  library {}
}

// -----

ssp.instance @deps_out_of_bounds of "Problem" {
  library {}
  graph {
    // expected-error @+1 {{Operand index is out of bounds for def-use dependence attribute}}
    "ssp.operation"() {dependences = [#ssp.dependence<2>]} : () -> ()
  }
}

// -----

ssp.instance @deps_defuse_not_increasing of "Problem" {
  library {}
  graph {
    %0:2 = ssp.operation<>()
    // expected-error @+1 {{Def-use operand indices in dependence attribute are not monotonically increasing}}
    "ssp.operation"(%0#0, %0#1) {dependences = [#ssp.dependence<1>, #ssp.dependence<0>]} : (none, none) -> ()
  }
}

// -----

ssp.instance @deps_interleaved of "Problem" {
  library {}
  graph {
    %0 = operation<> @Op()
    // expected-error @+1 {{Auxiliary dependence from @Op is interleaved with SSA operands}}
    operation<>(@Op, %0)
  }
}

// -----

ssp.instance @deps_aux_not_consecutive of "Problem" {
  library {}
  graph {
    operation<> @Op()
    // expected-error @+1 {{Auxiliary operand indices in dependence attribute are not consecutive}}
    "ssp.operation"() {dependences = [#ssp.dependence<1, @Op>]} : () -> ()
  }
}

// -----

ssp.instance @deps_aux_invalid of "Problem" {
  library {}
  graph {
    // expected-error @+1 {{Auxiliary dependence references invalid source operation: @InvalidOp}}
    operation<>(@InvalidOp)
  }
}

// -----

ssp.instance @linked_opr_invalid of "Problem" {
  library {}
  graph {
    // expected-error @+1 {{Linked operator type property references invalid operator type: @InvalidOpr}}
    operation<@InvalidOpr>()
  }
}

// -----

ssp.library @standalone {}
ssp.instance @standalone_opr_invalid of "Problem" {
  library {}
  graph {
    // expected-error @+1 {{Linked operator type property references invalid operator type: @standalone::@InvalidOpr}}
    operation<@standalone::@InvalidOpr>()
  }
}
