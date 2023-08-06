// RUN: circt-opt %s -ssp-roundtrip=verify -verify-diagnostics -split-input-file

// expected-error@+1 {{Invalid initiation interval}}
ssp.instance @no_II of "CyclicProblem" {
  library {}
  graph {}
}

// -----

// expected-error@+1 {{Precedence violated for dependence}}
ssp.instance @backedge_violated of "CyclicProblem" [II<2>] {
  library {
    operator_type @_1 [latency<1>]
  }
  graph {
    %0 = operation<@_1>(@op2 [dist<1>]) [t<0>]
    %1 = operation<@_1>(%0) [t<1>]
    operation<@_1> @op2(%1) [t<2>]
  }
}
