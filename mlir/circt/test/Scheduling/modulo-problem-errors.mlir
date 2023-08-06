// RUN: circt-opt %s -ssp-roundtrip=verify -verify-diagnostics -split-input-file

// expected-error@+1 {{Operator type 'limited' is oversubscribed}}
ssp.instance @oversubscribed of "ModuloProblem" [II<2>] {
  library {
    operator_type @limited [latency<1>, limit<2>]
  }
  graph {
    operation<@limited>() [t<1>]
    operation<@limited>() [t<3>]
    operation<@limited>() [t<5>]
  }
}
