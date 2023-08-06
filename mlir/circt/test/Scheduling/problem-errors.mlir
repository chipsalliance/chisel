// RUN: circt-opt %s -ssp-roundtrip=verify -verify-diagnostics -split-input-file

// expected-error@+1 {{Operator type 'foo' has no latency}}
ssp.instance @no_latency of "Problem" {
  library {
    operator_type @foo
  }
  graph {}
}

// -----

ssp.instance @no_starttime of "Problem" {
  library {
    operator_type @_1 [latency<1>]
  }
  graph {
    operation<@_1>() // expected-error {{Operation has no start time}}
  }
}

// -----

// expected-error@+1 {{Precedence violated for dependence}}
ssp.instance @ssa_dep_violated of "Problem" {
  library {
    operator_type @_1 [latency<1>]
  }
  graph {
    %0 = operation<@_1>() [t<0>]
    %1 = operation<@_1>(%0) [t<0>]
  }
}

// -----

// expected-error@+1 {{Precedence violated for dependence}}
ssp.instance @aux_dep_violated of "Problem" {
  library {
    operator_type @_1 [latency<1>]
  }
  graph {
    operation<@_1> @op0() [t<0>]
    operation<@_1> @op1(@op0) [t<0>]
  }
}
