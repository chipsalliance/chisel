// RUN: circt-opt %s -ssp-schedule=scheduler=asap -verify-diagnostics -split-input-file

// expected-error@+1 {{dependence cycle detected}}
ssp.instance @cyclic_graph of "Problem" {
  library {
    operator_type @_1 [latency<1>]
  }
  graph {
    %0 = operation<@_1>(@op2)
    %1 = operation<@_1>(%0)
    operation<@_1> @op2(%1)
  }
}
