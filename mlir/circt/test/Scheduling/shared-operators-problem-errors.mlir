// RUN: circt-opt %s -ssp-roundtrip=verify -verify-diagnostics -split-input-file

// expected-error@+1 {{Limited operator type 'limited' has zero latency}}
ssp.instance @limited_but_zero_latency of "SharedOperatorsProblem" {
  library {
    operator_type @limited [latency<0>, limit<1>]
  }
  graph {}
}

// -----

// expected-error@+1 {{Operator type 'limited' is oversubscribed}}
ssp.instance @oversubscribed of "SharedOperatorsProblem" {
  library {
    operator_type @limited [latency<1>, limit<2>]
  }
  graph {
    operation<@limited>() [t<0>]
    operation<@limited>() [t<0>]
    operation<@limited>() [t<0>]
  }
}
