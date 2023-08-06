// RUN: circt-opt %s -ssp-schedule=scheduler=simplex -verify-diagnostics -split-input-file

// expected-error@+1 {{problem is infeasible}}
ssp.instance @cyclic_graph of "CyclicProblem" {
  library {
    operator_type @_1 [latency<1>]
  }
  graph {
    %0 = operation<@_1>(@op2)
    %1 = operation<@_1>(%0)
    operation<@_1> @op2(%1)
  }
}
